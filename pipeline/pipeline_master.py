from nebula_api.milvus_api import connect_db
from nebula_api.scene_detector_api import NEBULA_SCENE_DETECTOR
import pickle

from posixpath import basename
import os
from pathlib import Path
import redis
import cv2 as cv
import pandas as pd

from experts.tracker import autotracker as at

def auxilary_function():
    """
    The function is for testing annd debugging
    :return:
    """
    db = connect_db('nebula_datadriven')
    query = 'FOR doc IN Movies RETURN doc'
    # query = 'FOR doc IN milvus_bert_embeddings RETURN doc'
    cursor = db.aql.execute(query, ttl=3600)
    name_arr = []
    for k, data in enumerate(cursor):
        name_arr.append('/movies/' + data['movie_name'] + '.avi')

    pass


class Pipeline:
    def __init__(self, debug=False):
        self.db_name = 'nebula_datadriven'
        # Removed from constructor due to memory issues
        # self.scene_detector = NEBULA_SCENE_DETECTOR()
        self.parent_folder = Path(__file__).parent.parent.parent.parent
        self.STEP_folder = os.path.join(self.parent_folder, 'STEP')
        self.debug = debug
        self.redis_cfg = {'host': '127.0.0.1', 'port': 6379}
        self.redis = redis.Redis(host=self.redis_cfg['host'], port=self.redis_cfg['port'], db=0)
        self.db = connect_db(self.db_name)

    def movie_exists(self, movie_path):
        # query = f'FOR doc IN Movies FILTER doc._id == \'{movie_id}\' RETURN doc'
        query = f'FOR doc IN Movies FILTER doc.full_path == \'{movie_path}\' RETURN doc'
        cursor = self.db.aql.execute(query, ttl=3600)
        for data in cursor:
            return True

        return False

    def divide_movie_into_frames(self, movie_in_path, movie_out_folder):
        cap = cv.VideoCapture(movie_in_path)
        ret, frame = cap.read()
        num = 0
        cv.imwrite(os.path.join(movie_out_folder, f'frame{num:04}.jpg'), frame)
        while cap.isOpened() and ret:
            num = num + 1
            ret, frame = cap.read()
            if frame is not None:
                cv.imwrite(os.path.join(movie_out_folder, f'frame{num:04}.jpg'), frame)
        return num

    def read_labelmap(self, labelmap_file):
        labelmap = []
        class_ids = set()
        name = ""
        class_id = ""
        for line in labelmap_file:
            if line.startswith("  name:"):
                name = line.split('"')[1]
            elif line.startswith("  id:") or line.startswith("  label_id:"):
                class_id = int(line.strip().split(" ")[-1])
                labelmap.append({"id": class_id, "name": name})
                class_ids.add(class_id)
        return labelmap, class_ids

    def step_results_postprocessing(self, results_path, lbl_file):
        print('STAG: postprocessing start')
        df = pd.read_csv(results_path, header=None,
                         names=['Dir', 'fileID', 'boxY_min', 'boxX_min', 'boxY_max', 'boxX_max', 'label', 'score'])

        # lbl_file = './stag/STEP/external/ActivityNet/Evaluation/ava/ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt'
        lf = open(lbl_file, 'r')
        r, _ = self.read_labelmap(lf)
        actions = {t['id']: t['name'] for t in r}
        df = df.replace({'label': actions})

        return df

    def insert_node_to_scenegraph(self, movie_id, arango_id, _class, scene_element, description, start, stop, bboxes):
        query = 'UPSERT { movie_id: @movie_id, description: @description, scene_element: @scene_element} INSERT  \
            { movie_id: @movie_id, arango_id: @arango_id, class: @class, description: @description, \
                 scene_element: @scene_element, start: @start, stop: @stop, step: 1, bboxes: @bboxes} UPDATE \
                { step: OLD.step + 1 } IN Nodes \
                    RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
        bind_vars = {'movie_id': movie_id,
                     'arango_id': arango_id,
                     'class': _class,
                     'scene_element': scene_element,
                     'start': start,
                     'stop': stop,
                     'description': description,
                     'bboxes': bboxes
                     }
        cursor = self.db.aql.execute(query, bind_vars=bind_vars, ttl=3600)
        for doc in cursor:
            doc = doc
        return doc['doc']['_id']

    def get_movie_scenes(self, movie_id):
        query = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}"  RETURN doc'.format(movie_id)
        cursor = self.db.aql.execute(query, ttl=3600)
        all_scenes = []
        for i, data in enumerate(cursor):
            # print("SCENE ELEM.: ", data['scene_'])
            scene = {'scene_graph_triplets': data['scene_graph_triplets'], 'movie_id': data['movie_id'],
                     'arango_id': data['arango_id'], 'description': data['description'],
                     'scene_element': data['scene_element'], 'start': data['start'], 'stop': data['stop']}
            # print(scene)
            all_scenes.append(scene)
        return all_scenes

    def upload_step_results(self, df: pd.DataFrame, movie_id):
        """
        Given movie_id (actually arango_id) and df from the STEP results, upload it to Arango
        :param df:
        :param movie_id:
        :return:
        """
        scenes = self.get_movie_scenes(movie_id)
        # Go over all rows from the DF, check to which scene element it belongs
        for k in range(df.shape[0]):
            scene_elem = -1
            for s in scenes:
                if df['fileID'][k] >= s['start'] and df['fileID'][k] < s['stop']:
                    scene_elem = s['scene_element']
                    start = s['start']
                    stop = s['stop']
                    arango_id = movie_id
                    description = df['label'][k]
                    _class = 'Object'
                    bbox = [df['boxX_min'][k], df['boxY_min'][k], df['boxX_max'][k], df['boxY_max'][k]]
                    break
            if scene_elem >= 0:
                self.insert_node_to_scenegraph(movie_id, arango_id, _class, scene_elem, description, start, stop, bbox)

    def insert_movie(self, movie_path):
        file_name = basename(movie_path)
        movie_name = file_name.split('.')[0]
        if not self.debug:
            scene_detector = NEBULA_SCENE_DETECTOR()
            movie_id = scene_detector.insert_movie(file_name, movie_name,
                                                              ["hollywood", "pegasus", "visual genome"], movie_path,
                                                              "media/videos/" + file_name, 30000)
        else:
            movie_id = '0'

        return movie_id


    def run_tracker_on_video(self):
        pass


    def process_new_movie(self, movie_path):
        # 1. Check if the movie exists
        if not self.debug and self.movie_exists(movie_path):
            return

        # 2. detect scenes and insert it into our database
        movie_id = self.insert_movie(movie_path)

        # 3. Divide the movie into frames and save them in redis
        frames_folder = os.path.join(self.parent_folder, 'frames/')
        if not os.path.exists(frames_folder):
            os.mkdir(frames_folder)
        else:
            for f in os.listdir(frames_folder):
                if os.path.isfile(os.path.join(frames_folder, f)):
                    os.remove(os.path.join(frames_folder, f))
        num_frames = self.divide_movie_into_frames(movie_path, frames_folder)

        # SAVE TO REDIS - TBD
        if num_frames > 0:
            img_list = []
            for k in range(num_frames):
                img_name = os.path.join(frames_folder, f'frame{k:04}.jpg')
                img = cv.imread(img_name)
                img_list.append(img)
            img_list_serial = pickle.dumps(img_list)
            self.redis.mset({movie_id: img_list_serial})

                # val_bin = pickle.dumps(val)
                # redis_key = video_name + ':' + slice_name
                # assert self.redis.set(redis_key, val_bin)


        # 4. Apply STEP
        cmd_line = '/home/ec2-user/miniconda3/envs/michael/bin/python ' + os.path.join(self.STEP_folder,
                'demo.py') + ' --input_folder ' + os.path.join(self.parent_folder, 'nebula/')
        os.system(cmd_line)

        # 5. Postprocess STEP results
        df = self.step_results_postprocessing(os.path.join(self.parent_folder, 'results/results.txt'),
            self.STEP_folder + '/external/ActivityNet/Evaluation/ava/ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt')

        self.upload_step_results(df, movie_id)

if __name__ == '__main__':
    print('Start Pipeline example')
    # auxilary_function()

    # We assume that we get a movie id
    pipeline = Pipeline(debug=True)
    movie_path = '/movies/actioncliptrain00188.avi'
    # movie_path = '/movies/actioncliptrain00188_new.avi'
    pipeline.process_new_movie(movie_path)

    print('End Pipeline example')