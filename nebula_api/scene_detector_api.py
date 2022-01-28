from posixpath import basename
from typing import Counter
from scenedetect.video_splitter import split_video_ffmpeg, is_ffmpeg_available
# Standard PySceneDetect imports:
from scenedetect import VideoManager, scene_detector
from scenedetect import SceneManager
# For content-aware scene detection:
from scenedetect.detectors import ContentDetector
from pickle import STACK_GLOBAL
import cv2
import os
import logging
import uuid
from arango import ArangoClient
from benchmark.clip_benchmark import NebulaVideoEvaluation
import numpy as np
from nebula_api.nebula_enrichment_api import NRE_API
import glob
import redis
import boto3
import json
import csv

class NEBULA_SCENE_DETECTOR():
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            level=logging.INFO)
        #self.connect_db('nebula_development')
        self.video_eval = NebulaVideoEvaluation()
        self.nre = NRE_API()
        self.db = self.nre.db
        self.redis_cfg = {'host': 'mainnode', 'port': 6379}
        self.redis = redis.Redis(
            host=self.redis_cfg['host'], port=self.redis_cfg['port'], db=0)
        self.s3 = boto3.client('s3', region_name='eu-central-1')

    # def init_new_db(self, dbname):
    #     client = ArangoClient(hosts='http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:8529')
    #     sys_db = client.db('_system', username='root', password='nebula')

    #     if not sys_db.has_database(dbname):
    #         sys_db.create_database(dbname,users=[{'username': 'nebula', 'password': 'nebula', 'active': True}])

    #     db = client.db(dbname, username='nebula', password='nebula')
        
    #     #StoryLine
    #     if db.has_collection('StoryLine'):
    #         ssn = db.collection('StoryLine')
    #     else:
    #         ssn = db.create_collection('StoryLine')

    #     if db.has_collection('Nodes'):
    #         ssn = db.collection('Nodes')
    #     else:
    #         ssn = db.create_collection('Nodes')

    #     if db.has_collection('Edges'):
    #         sse = db.collection('Edges')
    #     else:
    #         sse = db.create_collection('Edges',  edge=True)
        
    #     if db.has_collection('MovieToStory'):
    #         sse = db.collection('MovieToStory')
    #     else:
    #         sse = db.create_collection('MovieToStory',  edge=True)

    #     if db.has_graph('StoryGraph'):
    #         nebula_graph_storyg = db.graph('StoryGraph')
    #     else:
    #         nebula_graph_storyg = db.create_graph('StoryGraph')

    #     if not nebula_graph_storyg.has_edge_definition('Edges'):
    #         actors2asset = nebula_graph_storyg.create_edge_definition(
    #             edge_collection='Edges',
    #             from_vertex_collections=['Nodes'],
    #             to_vertex_collections=['Nodes']
    #         )
        
    #     if not nebula_graph_storyg.has_edge_definition('MovieToStory'):
    #         movie2actors = nebula_graph_storyg.create_edge_definition(
    #             edge_collection='MovieToStory',
    #             from_vertex_collections=['Movies'],
    #             to_vertex_collections=['Nodes']
    #         )

    def detect_scene_elements(self, video_file):
        print("DEBUG: ", video_file)
        scenes = []
        video_manager = VideoManager([video_file])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))
    # Improve processing speed by downscaling before processing.
        video_manager.set_downscale_factor()
    # Start the video manager and perform the scene detection.
        video_manager.start()     
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            stop_frame = scene[1].get_frames()
            scenes.append([start_frame,stop_frame])
        print("Scenes: ", scenes)
        return(scenes)

    def divide_movie_into_frames(self, movie_in_path, movie_out_folder):
        cap = cv2.VideoCapture(movie_in_path)
        ret, frame = cap.read()
        num = 0
        cv2.imwrite(os.path.join(movie_out_folder, f'frame{num:04}.jpg'), frame)
        while cap.isOpened() and ret:
            num = num + 1
            ret, frame = cap.read()
            if frame is not None:
                cv2.imwrite(os.path.join(movie_out_folder,
                           f'frame{num:04}.jpg'), frame)
        return num

    def store_frames_to_s3(self, movie_id, frames_folder, video_file):
        bucket_name = "nebula-frames"
        folder_name = movie_id
        self.s3.put_object(Bucket=bucket_name, Key=(folder_name+'/'))
        print(frames_folder)
        if not os.path.exists(frames_folder):
            os.mkdir(frames_folder)
        else:
            for f in os.listdir(frames_folder):
                if os.path.isfile(os.path.join(frames_folder, f)):
                    os.remove(os.path.join(frames_folder, f))
        num_frames = self.divide_movie_into_frames(video_file, frames_folder)
        # SAVE TO REDIS - TBD
        if num_frames > 0:
            for k in range(num_frames):
                img_name = os.path.join(
                    frames_folder, f'frame{k:04}.jpg')
                self.s3.upload_file(img_name, bucket_name, folder_name +
                            '/' + f'frame{k:04}.jpg')


    def get_video_metadata(self, video_file):
        cap = cv2.VideoCapture(video_file)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return ({'width': W, 'height': H, 'fps': fps})

    def detect_scenes(self, video_file):
        scenes = []
        
        video_manager = VideoManager([video_file])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=70.0))
    # Improve processing speed by downscaling before processing.
        video_manager.set_downscale_factor()
    # Start the video manager and perform the scene detection.
        video_manager.start()     
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            stop_frame = scene[1].get_frames()
            scenes.append([start_frame,stop_frame])
        print("Scenes: ", scenes)
        return(scenes)
    
    def detect_mdf(self, video_file, scene_elements, method='3frames'):
        """
        :param video_file:
        :param scene_elements: list of list, [[2, 4], [33, 55]] - start and end frame of each scene
        :param method: '3frames' - choose three good frames, 'clip_segment' - use clip segmentation to choose 3 MDFs
        :return:
        """
        print("TODO")
        if method == '3frames':
            mdfs = []
            for scene_element in scene_elements:
                scene_mdfs = []
                start_frame = scene_element[0]
                stop_frame = scene_element[1]
                frame_qual = self.video_eval.mark_blurred_frames(video_file, start_frame, stop_frame, 100)
                # Ignore the blurred images
                frame_qual[0:3] = 0
                frame_qual[-3:] = 0
                middle_frame = start_frame + ((stop_frame - start_frame) // 2)
                good_frames = np.where(frame_qual > 0)[0]
                if len(good_frames > 5):
                    stop_frame = start_frame + good_frames[-1]
                    start_frame = start_frame + good_frames[0]
                    middle_frame = start_frame + good_frames[len(good_frames) // 2]
                    scene_mdfs.append(int(start_frame))
                    scene_mdfs.append(int(middle_frame))
                    scene_mdfs.append(int(stop_frame))
                else:
                    scene_mdfs.append(int(start_frame) + 2)
                    scene_mdfs.append(int(middle_frame))
                    scene_mdfs.append(int(stop_frame) - 2)
                mdfs.append(scene_mdfs)
            # go over all frames and compute the average derivative
        elif method == 'clip_segment':
            pass
        else:
            raise Exception('Unsupported method')
        # mdfs = []
        # for scene_element in scene_elements:
        #     scene_mdfs = []
        #     start_frame = scene_element[0]
        #     stop_frame = scene_element[1]
        #     scene_mdfs.append(start_frame + 2)
        #     middle_frame = start_frame + ((stop_frame- start_frame) // 2)
        #     scene_mdfs.append(middle_frame)
        #     scene_mdfs.append(stop_frame - 2)
        #     mdfs.append(scene_mdfs)
        return(mdfs)


    #self.nre.update_expert_status("movies")
    def convert_avi_to_mp4(self, avi_file_path, output_name):
        os.system("ffmpeg -y -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(
            input=avi_file_path, output=output_name))
        return True

    # file_name - file name without path
    # movie_name - if available, if no - same file name
    # tags - if available
    # full_path - path to file, for video processing
    # url - url to S3
    # last_frame - number of frames in movie
    # metadata - available metadata - fps, resolution....
    def insert_movie(self, file_name, movie_name, tags, full_path, url, last_frame, status):
        movie_id = uuid.uuid4().hex
        query = 'UPSERT { movie_name: @movie_name} INSERT  \
            { movie_id: @movie_id, file_name: @file_name, movie_name: @movie_name, description: @description, tags: @tags,\
                full_path: @full_path, url_path: @url, last_frame: @last_frame,\
                scenes: @scenes, scene_elements: @scene_elements, mdfs: @mdfs, meta: @metadata,\
                    last_frame: @last_frame, scenes: @scenes, scene_elements: @scene_elements, mdfs: @mdfs, updates: 1, \
                         status: "created"\
                    } UPDATE \
                { updates: OLD.updates + 1, file_name: @file_name, description: @description, tags: @tags,\
                full_path: @full_path, url_path: @url, last_frame: @last_frame,\
                scenes: @scenes, scene_elements: @scene_elements, mdfs: @mdfs, meta: @metadata, status: @status, \
                    last_frame: @last_frame, scenes: @scenes, scene_elements: @scene_elements, mdfs: @mdfs \
                } IN Movies \
                    RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
        scene_elements = self.detect_scene_elements(full_path)
        bind_vars = {
                        'movie_id': movie_id,
                        'file_name': file_name,
                        'movie_name': movie_name,
                        'description': movie_name,
                        'tags': tags,
                        'full_path': full_path,
                        'url': url,
                        'last_frame': last_frame,
                        'scenes': self.detect_scenes(full_path),
                        'scene_elements': scene_elements,
                        'mdfs': self.detect_mdf(full_path, scene_elements),
                        'metadata': self.get_video_metadata(full_path),
                        'status': status
                        }
        print(bind_vars)
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        for doc in cursor:
            doc=doc
        return(doc['doc']['_id'])

    def new_movies_batch_processing(self, upload_dir, storage_dir, dataset):
        _files = glob.glob(upload_dir + '/*')
        movies = []
        for _file in _files:
            file_name = basename(_file)
            movie_mame = file_name.split(".")[0]            
            print("New movie found")
            print(file_name)
            print(movie_mame)
            self.convert_avi_to_mp4(_file, storage_dir + "/" + movie_mame)
            movie_id = self.insert_movie(file_name, movie_mame, ["lsmdc", "pegasus", "visual genome"],
                              storage_dir + "/" + movie_mame + ".mp4", "static/" + dataset + "/" + movie_mame + ".mp4", 300, "created")
            movies.append((_file, movie_id))
        return(movies)
    
    def merge_movies(self, upload_dir, storage_dir):
        print("Source directory: {}".format(upload_dir))
        print("Destination directory: {}".format(storage_dir))

        _files = glob.glob(upload_dir + '/*')
        movie_names = ['0001_American_Beauty']
        movies_full_path = []
        for movie_name in movie_names:
            movies_full_path.append(os.path.join(upload_dir, movie_name))
        
        # Merge movie clips of movie into one movie
        for _file in _files:
            if _file in movies_full_path:
                movie_clips = sorted(os.listdir(_file))

                 # Create a new video
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                movie_name = _file.split("/")[-1]
                clip_metadata = self.get_video_metadata(os.path.join(_file, movie_clips[0]))
                movie_name_path = os.path.join(storage_dir, movie_name + ".avi")
                video = cv2.VideoWriter(movie_name_path, fourcc, clip_metadata['fps'], (clip_metadata['width'], clip_metadata['height']))

                print("Working on movie: " + str(movie_name))

                # Iterate over the clips of the movie
                for clip in movie_clips:
                    clip = os.path.join(_file, clip)
                    curr_v = cv2.VideoCapture(clip)
                    while curr_v.isOpened():
                        r, frame = curr_v.read()  # Get return value and curr frame of curr video
                        if not r:
                            break
                        video.write(frame)  # Write the frame

                video.release()  # Save the video
                print("Saved video to: {}".format(movie_name_path))

    def get_frame_dict(self, key_dic):

        # Define variables for parsed dictionary
        new_key = key_dic['img_fn'].split("/")[1].split("@")[0]
        fn = key_dic['img_fn'].split("/")[1].split("@")[1].split(".")[0]
        split = key_dic['split'] if 'split' in key_dic else ''
        place = key_dic['place'] if 'place' in key_dic else ''
        event = key_dic['event'] if 'event' in key_dic else ''
        intent = key_dic['intent'] if 'intent' in key_dic else ''
        before = key_dic['before'] if 'before' in key_dic else ''
        after = key_dic['after'] if 'after' in key_dic else ''

        processed_dict = {
            'clip_id': new_key,
            'fn': fn,
            'split': split,
            'place': place,
            'event': event,
            'intent': intent,
            'before': before,
            'after': after
        }
        return processed_dict

    def preprocess_dataset(self, dataset_path, db_type="COMET"):
        
        new_db = []
        if db_type == "COMET":
            test_json = open(dataset_path)
            data = json.load(test_json)
            for idx, key in enumerate(data):
                if "lsmdc" in key['img_fn']:
                    new_key = self.get_frame_dict(key)
                    new_db.append(new_key)

        elif db_type == "LSMDC":
            with open(dataset_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    key = row[0].split()[0]
                    txt = ''
                    if len(row) > 1:
                        txt = row[1]
                    new_db.append({'clip_id': key, 'txt': txt})
            return new_db
        else: #db_type is LSMDC
            print("Unknown DB-TYPE")

        return new_db

    # Merge all annotations for the same frame in VisualCOMET
    def merge_clip_ids(self, dataset):
        # counter = 0
        new_dataset = []

        prev_clip = dataset[0]

        for i in range(1, len(dataset)):
            cur_clip = dataset[i]

            if (isinstance(prev_clip, list) and prev_clip[0]['clip_id'] != cur_clip['clip_id']) or \
                (isinstance(prev_clip, dict) and prev_clip['clip_id'] != cur_clip['clip_id']):
                    new_dataset.append(dataset[i-1])

            # If we have multiple dicts, then we have different frames so we store it in list of dics
            # and we want to make sure we have same clip_ids in the current dics we're comparing
            # We also the now atleast a list of two dicts.
            if isinstance(prev_clip, list) and prev_clip[0]['clip_id'] == cur_clip['clip_id']:
                # Check if we have the cur_clip's fn already in the list
                # If yes, we add all the information to that fn instead of expending the list with similar fn's.
                
                found_same_fn = False
                for obj in prev_clip:
                    if obj['clip_id'] == cur_clip['clip_id'] and \
                        obj['fn'] == cur_clip['fn']:
                        found_same_fn = True
                        if obj['place'] != cur_clip['place']:
                            obj['place'] += (", " + cur_clip['place'])

                        if obj['event'] != cur_clip['event']:
                            obj['event'] += (", " +cur_clip['event'])

                        if obj['intent'] != cur_clip['intent']:
                            obj['intent'] += cur_clip['intent']
                        
                        if obj['before'] != cur_clip['before']:
                            obj['before'] += cur_clip['before']
                        
                        if obj['after'] != cur_clip['after']:
                            obj['after'] += cur_clip['after']
                        
                        # counter += 1

                # If we don't have the current fn in our list of dics, so we add the current dict with that new frame
                if not found_same_fn:
                    prev_clip.append(cur_clip)
                
                dataset[i] = prev_clip
                        
            else:
                if isinstance(prev_clip, list):
                    prev_clip_id = prev_clip[0]['clip_id']
                else:
                    prev_clip_id = prev_clip['clip_id']
            
                if i > 0 and prev_clip_id == cur_clip['clip_id']:
                    # If we have multiple frames on same timestamp
                    # We add the two dicts to a list.
                    if prev_clip['fn'] != cur_clip['fn']:
                        temp_lst = []
                        temp_lst.append(cur_clip)
                        temp_lst.append(prev_clip)
                        cur_clip = temp_lst

                    else:
                        if prev_clip['place'] != cur_clip['place']:
                            cur_clip['place'] += (", " + prev_clip['place'])

                        if prev_clip['event'] != cur_clip['event']:
                            cur_clip['event'] += (", " +prev_clip['event'])

                        if prev_clip['intent'] != cur_clip['intent']:
                            cur_clip['intent'] += prev_clip['intent']
                        
                        if prev_clip['before'] != cur_clip['before']:
                            cur_clip['before'] += prev_clip['before']
                        
                        if prev_clip['after'] != cur_clip['after']:
                            cur_clip['after'] += prev_clip['after']

                        # counter += 1

                prev_clip = cur_clip
                dataset[i] = prev_clip

        # print(counter)
        return new_dataset


    def merge_comet_in_lsmdc(self, lsmdc_datasets, comet_dataset):
        found_keys = 0

        temp_comet_dataset = []
        merged_dbs = []
        # Converting list to dict so we can check keys in O(1)
        # comet_ds = Counter(comet_dataset['clip_id'])

        for lsmdc_ds in lsmdc_datasets:
            lsmdc_ds_clip_ids_dict = {k['clip_id']: idx for idx, k in enumerate(lsmdc_ds)}
            for idx, comet_key in enumerate(comet_dataset):
                if isinstance(comet_key, list):
                    cur_comet_key = comet_key[0]['clip_id']
                    # found_keys += len(comet_key) - 1
                else:
                    cur_comet_key = comet_key['clip_id']

                if cur_comet_key in lsmdc_ds_clip_ids_dict:
                    found_keys += 1
                    if isinstance(comet_key, list):
                        temp_comet_dataset.append(comet_key)
                    else:
                        temp_comet_dataset.append(comet_dataset[idx])
                    
                    index_in_lsmdc = lsmdc_ds_clip_ids_dict[cur_comet_key]
                    lsmdc_clip_details = lsmdc_ds[index_in_lsmdc]
                    vscomet_clip_details = comet_key
                    merged_dbs.append({'lsmdc': [lsmdc_clip_details],
                                        'vscomet': [vscomet_clip_details]})

        
        return merged_dbs

    def parse_comet_to_lsmdc(self, comet_dir, lsmdc_dir):

        ### VISUAL COMET ###

        # Paths to VISUAL COMET .JSONs
        comet_train_path = os.path.join(comet_dir, "train_annots.json")
        comet_val_path = os.path.join(comet_dir, "val_annots.json")
        comet_test_path = os.path.join(comet_dir, "test_annots.json")

        # Parsed COMET .JSONs for LSMDC dataset
        comet_train_dict = self.preprocess_dataset(comet_train_path, "COMET")
        comet_val_dict = self.preprocess_dataset(comet_val_path, "COMET")
        comet_test_dict = self.preprocess_dataset(comet_test_path, "COMET")

        # Sort COMET JSONs
        comet_train_dict = sorted(comet_train_dict, key=lambda d: d['clip_id'])
        comet_val_dict = sorted(comet_val_dict, key=lambda d: d['clip_id'])
        comet_test_dict = sorted(comet_test_dict, key=lambda d: d['clip_id'])

        # Merge all annotations for the same frame
        comet_train_dict = self.merge_clip_ids(comet_train_dict)
        comet_val_dict = self.merge_clip_ids(comet_val_dict)
        comet_test_dict = self.merge_clip_ids(comet_test_dict)

        ### LSMDC ###

        # Paths to LSMDC .JSONs
        lsmdc_train_path = os.path.join(lsmdc_dir, "LSMDC16_annos_training.csv")
        lsmdc_val_path = os.path.join(lsmdc_dir, "LSMDC16_annos_val.csv")
        lsmdc_test_path = os.path.join(lsmdc_dir, "LSMDC16_annos_test.csv")

        # Process LSMDC annotation files
        lsmdc_train_dict = self.preprocess_dataset(lsmdc_train_path, "LSMDC")
        lsmdc_val_dict = self.preprocess_dataset(lsmdc_val_path, "LSMDC")
        lsmdc_test_dict = self.preprocess_dataset(lsmdc_test_path, "LSMDC")

        # find COMET's clips in LSMDC dataset 
        lsmdc_datasets = [lsmdc_train_dict, lsmdc_val_dict, lsmdc_test_dict]
        parsed_train_path = self.merge_comet_in_lsmdc(lsmdc_datasets, comet_train_dict)
        parsed_val_path = self.merge_comet_in_lsmdc(lsmdc_datasets, comet_val_dict)
        parsed_test_path = self.merge_comet_in_lsmdc(lsmdc_datasets, comet_test_dict)
        
def divide_movie_by_timestamp(movie_in_path, t_start, t_end, dim=(224, 224)):

    cap = cv2.VideoCapture(movie_in_path)
    ret, frame = cap.read()
    num = 0
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    movie_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_start = t_start * fps
    frame_end = t_end * fps
    if frame_start > movie_frames_count or frame_end > movie_frames_count:
        print(f'Error, the provided t_start {t_start} or t_end {t_end} \
            multiplied by fps {fps} is higher than number of frames {movie_frames_count}')
        return -1
    while cap.isOpened() and ret:
        num = num + 1
        ret, frame = cap.read()
        if frame is not None and num >= frame_start and num <= frame_end:
            resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            frames.append(resized_frame)
    return frames






def main():
    scene_detector = NEBULA_SCENE_DETECTOR()
    # scene_detector.parse_comet_to_lsmdc("/dataset1/visualcomet/", "/dataset1/")
    # scene_detector.merge_movies("/dataset1/", "/dataset1/storage/")
    path = '/dataset/development/1010_TITANIC_00_41_32_072-00_41_40_196.mp4'
    path_frames = '/home/ilan/development/frames/'
    divide_movie_by_timestamp(path, 3, 4)

    # scene_detector.new_movies_batch_processing()

    #scene_detector.init_new_db("nebula_datadriven")
    # _files = glob.glob('/movies/*avi')
    # #Example usage
    # for _file in _files:
    #     file_name = basename(_file)
    #     movie_mame = file_name.split(".avi")[0]
    #     scene_detector.insert_movie(file_name, movie_mame, ["hollywood", "pegasus", "visual genome"],
    #     _file,"static/datadriven/" + file_name, 300)
if __name__ == "__main__":
    main()


    
