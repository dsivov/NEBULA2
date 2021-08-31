import logging
import clip
import torch
import cv2
import math
import glob
import os
import numpy as np
from scenedetect import VideoManager
from scenedetect import SceneManager
# For content-aware scene detection:
from scenedetect.detectors import ContentDetector
from arango import ArangoClient
from nebula_api.milvus_api import MilvusAPI
from pathlib import Path
from PIL import Image
from nebula_api.graph_encoder import GraphEncoder
from nebula_api.nebula_enrichment_api import NRE_API


class STORE_SCENE:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            level=logging.INFO)


        # Choose device and load the chosen model
        #self.collection_name = 'nebula_clip'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("RN50x4", self.device)
        #self.connect_db('nebula_development')    
        self.scenes = MilvusAPI('milvus', 'scenes_hollywood_mean', 'nebula_development', 640) 
        self.bert_story = MilvusAPI(
            'milvus', 'bert_embeddings_development', 'nebula_development', 768)
        self.nre = NRE_API()
        self.db = self.nre.db
    
    def _calculate_images_features(self, frame):
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #image = Image.fromarray(image)
            image = self.preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
            return image_features
        else: 
            return None

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
        #print(len(new_features.tolist()[0]))
        self.scenes.insert_vectors([new_features.tolist()[0]], [meta])
        #input()
    
    def delete_all_vectors(self, movie):
        self.scenes.delete_movie_vector(movie, "bert_embeddings_development")
        self.scenes.delete_movie_vector(movie, "scenes_hollywood_mean")

    def get_movies(self):
        #query = 'FOR doc IN Movies RETURN doc'
        query = 'FOR doc IN Movies RETURN doc'
        nebula_movies = []
        cursor = self.db.aql.execute(query)
        for data in cursor: 
            nebula_movies.append(data)
        return(nebula_movies) 

    def get_stages(self, m):
        query_r = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}" RETURN doc'.format(m)
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
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + 1)
                ret, frame_f = cap.read() # Read the frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                ret, frame_m = cap.read() # Read the frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, stop_frame - 1)
                ret, frame_l = cap.read() # Read the frame
                imf = []
                if not ret:
                    print("Frame not found ")
                else:
                    feature_f = self._calculate_images_features(frame_f)
                    feature_m = self._calculate_images_features(frame_m)
                    feature_l = self._calculate_images_features(frame_l)    
                    if torch.is_tensor(feature_f):
                        imf.append(feature_f.cpu().detach().numpy())
                    if torch.is_tensor(feature_m):
                        imf.append(feature_m.cpu().detach().numpy())
                    if torch.is_tensor(feature_l):
                        imf.append(feature_l.cpu().detach().numpy())
                    if len(imf) > 0:
                        feature_mean = np.mean(imf, axis=0)
                        feature_t = torch.from_numpy(feature_mean)  
                        
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
    
    def encode_movie_to_bert(self, movie_id):
        """
        Encode all scene elements of a movie
        :param movie_arango_id: for example, 'Movies/92349435'
        :param db_name: database name (nebula_datadriven)
        :return: torch.Size([1, 768]) + dictionary of meta data
        """
        query = f'FOR doc IN StoryLine FILTER doc.arango_id == \'{movie_id}\' RETURN doc'
        cursor = self.db.aql.execute(query, ttl=3600)
        cur_sentences = []
        graph_encoder = GraphEncoder()
        for cnt, sent_array in enumerate(cursor):
            cur_sentences.append(sent_array)
            meta = {
                'filename': 'none',
                'movie_id': sent_array['arango_id'],
                'nebula_movie_id': sent_array['arango_id'],
                'stage': sent_array['scene_element'],
                'frame_number': 'none',
                'sentence': sent_array['sentences'],
            }
        final_sent = graph_encoder.process_scene_sentences(cur_sentences)

        if final_sent is not None:
            embedding = graph_encoder.encode_with_transformers(final_sent)
            meta['sentence'] = final_sent
            self.bert_story.insert_vectors(embedding.tolist(), [meta])
            print(embedding.tolist(), [meta])

    def create_clip_scene(self, movie):
        stages = self.get_stages(movie)
        for stage in stages:
            print("Calculate scene for: ", stage['arango_id'])
            self.add_nebula_video_scene_detect(stage['full_path'], stage['arango_id'], stage['scene_element'], stage['start'], stage['stop'])



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
        file_name = "/movies/" + movie['file_name']
        print(file_name)
        stages = scenes.get_stages(movie['_id'])
        for stage in stages:
            print("Calculate scene for: ", stage['arango_id'])
            scenes.add_nebula_video_scene_detect(file_name, stage['arango_id'], stage['scene_element'], stage['start'], stage['stop'])
if __name__ == "__main__":
    main()
