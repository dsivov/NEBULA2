import logging
import clip
import torch
import cv2
import math
import glob
import os
from scenedetect import VideoManager
from scenedetect import SceneManager
# For content-aware scene detection:
from scenedetect.detectors import ContentDetector
from arango import ArangoClient
from nebula_api.milvus_api import MilvusAPI
from pathlib import Path
from PIL import Image

class STORE_SCENE:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            level=logging.INFO)


        # Choose device and load the chosen model
        #self.collection_name = 'nebula_clip'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("RN50x4", self.device)
        self.connect_db('nebula_dev')    
        self.scenes = MilvusAPI('milvus', 'scenes919', 'nebula_dev', 640) 

    def connect_db(self, dbname):
        #client = ArangoClient(hosts='http://ec2-18-219-43-150.us-east-2.compute.amazonaws.com:8529')
        #client = ArangoClient(hosts='http://35.158.120.92:8529')
        client = ArangoClient(hosts='http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:8529')
        db = client.db(dbname, username='nebula', password='nebula')
        self.db = db

    def _calculate_images_features(self, frame):
        image = self.preprocess(Image.fromarray(frame)).unsqueeze(0).to(self.device)
        #image_test = self.preprocess_test(Image.fromarray(frame)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            #image_features_test = self.model_test.encode_image(image_test)
        #print(image_features)
        return image_features

    def _add_image_features(self, new_features, feature_video_map):
        # Controls the addition of image features to the object
        # such that video mappings are provided, etc.,
        new_features /= new_features.norm(dim=-1, keepdim=True)
        meta = {
                            'filename': feature_video_map['video_path'],
                            'movie_id': feature_video_map['movie_id'],
                            'nebula_movie_id': feature_video_map['movie_id'],
                            'stage': feature_video_map['scene_id'],
                            'frame_number':feature_video_map['frame_id'],
                            'sentence': str(feature_video_map['start_frame_id']) + "-" + str(feature_video_map['stop_frame_id'])
                        }
        
        #print(new_features.tolist()[0])
        self.scenes.insert_vectors([new_features.tolist()[0]], [meta])
    
    def get_movies(self):
        #query = 'FOR doc IN Movies RETURN doc'
        query = 'FOR doc IN Movies FILTER doc.split == \'0\' AND doc.splits_total == \'30\' RETURN doc'
        nebula_movies = []
        cursor = self.db.aql.execute(query)
        for data in cursor: 
            nebula_movies.append(data)
        return(nebula_movies) 

    def get_stages(self, m):
        query_r = 'FOR doc IN SemanticStoryNodes FILTER doc.arango_movie == "{}"  AND doc.description == \'With\' RETURN doc'.format(m)
        cursor_r = self.db.aql.execute(query_r)
        stages = []
        for stage in cursor_r:
            stages.append(stage)
        return(stages)        
    
    def add_nebula_video_scene_detect(self, fn, movie_id, scene, start_frame, stop_frame):        
        if (fn):
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            if video_file.is_file():
                cap = cv2.VideoCapture(fn)
                print("Scene: ", scene )
                middle_frame = start_frame + ((stop_frame- start_frame) // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                ret, frame_ = cap.read() # Read the frame
                if not ret:
                    print("File not found")
                else:
                    feature_t = self._calculate_images_features(frame_)                 
                    feature_data = {'video_path': file_name,
                                    'start_frame_id': start_frame,
                                    'stop_frame_id': stop_frame,
                                    'scene_id': scene,
                                    'frame_id': middle_frame,
                                    'movie_id' : movie_id
                                    #'frame': frame_.tolist()
                                    }
                    self._add_image_features(feature_t, feature_data) 
               
            cap.release()
            cv2.destroyAllWindows()
            cap.release()
            cv2.destroyAllWindows()
                    #print (start_frame)                     
        else:
            print("File doesn't exist: ", fn)

def main():
    # clip = STORE_SCENE()
    # movies = glob.glob('/movies/*.avi') 
    # for movie in movies:    
    #     print("Processing Movie: ", movie)
    #     clip.add_nebula_video_scene_detect(movie)
    scenes = STORE_SCENE()
    movies = scenes.get_movies()
    for movie in movies:
        #file_name = "/movies/byscene/" + movie['url_path'].split("/splits/")[1]
        file_name = "/movies/" + movie['file_name'] + ".avi"
        print(file_name)
        stages = scenes.get_stages(movie['_id'])
        for stage in stages:
            scenes.add_nebula_video_scene_detect(file_name, stage['arango_movie'], stage['stage'], stage['start'], stage['stop'])
if __name__ == "__main__":
    main()
