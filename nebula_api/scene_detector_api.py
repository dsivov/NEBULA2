
from posixpath import basename
from scenedetect.video_splitter import split_video_ffmpeg, is_ffmpeg_available
# Standard PySceneDetect imports:
from scenedetect import VideoManager, scene_detector
from scenedetect import SceneManager
# For content-aware scene detection:
from scenedetect.detectors import ContentDetector
import os
from pickle import STACK_GLOBAL
import cv2
import logging
import uuid
from arango import ArangoClient
from benchmark.clip_benchmark import NebulaVideoEvaluation
import numpy as np
from nebula_api.nebula_enrichment_api import NRE_API
import glob
import redis
import boto3

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

def main():
   
    scene_detector = NEBULA_SCENE_DETECTOR()
    scene_detector.new_movies_batch_processing()
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


    
