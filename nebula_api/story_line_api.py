import os
from pickle import STACK_GLOBAL
import clip
import milvus
import torch
import cv2
import math
import glob
import logging
import time
from pathlib import Path


from PIL import Image
import numpy as np
from patchify import patchify
from nebula_api.milvus_api import MilvusAPI
from nebula_api.nebula_enrichment_api import NRE_API



class STORY_LINE_API:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            level=logging.INFO)


        # Choose device and load the chosen model
        self.collection_name = 'nebula_clip'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("RN50x4", self.device)
        #self.model_test, self.preprocess_test = clip.load("RN101", self.device)
        self.scene_graph = MilvusAPI('milvus', 'scene_graph_visual_genome', 'nebula_dev', 640) 
        self.pegasus_stories =  MilvusAPI('milvus', 'pegasus', 'nebula_dev', 640)
        self.nre = NRE_API()
        self.db = self.nre.db
        
    def get_all_movies_meta(self):
        nebula_movies={}
        #query = 'FOR doc IN Movies FILTER doc.split == \'0\' AND doc.splits_total == \'30\' RETURN doc'
        query = 'FOR doc IN Movies FILTER doc.status != "complited" RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/10715274\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/17342682\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/12911567\' RETURN doc'

        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/11723602\' RETURN doc'
        cursor = self.db.aql.execute(query)
        for data in cursor: 
            nebula_movies[data['_id']] = data
        return(nebula_movies) 

    def get_movie_meta(self, movie_id):
        nebula_movies={}
        #print(movie_id)
        #query = 'FOR doc IN Movies FILTER doc.split == \'0\' AND doc.splits_total == \'30\' RETURN doc'
        query = 'FOR doc IN Movies FILTER doc._id == "{}" AND doc.status != "complited" RETURN doc'.format(movie_id)
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/10715274\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/17342682\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/12911567\' RETURN doc'

        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/11723602\' RETURN doc'
        cursor = self.db.aql.execute(query)
        for data in cursor:
            #print(data) 
            nebula_movies[data['_id']] = data
        return(nebula_movies) 

    def insert_scene_to_storyline(self, file_name,  movie_id, arango_id ,scene_element, mdf, description, start, stop, sentences, triplets):
        query = 'UPSERT { movie_id: @movie_id, scene_element: @scene_element} INSERT  \
            { movie_id: @movie_id, arango_id: @arango_id, description: @description, full_path: @file_name, scene_element: @scene_element, mdfs: @mdf, start: @start\
                , stop: @stop, sentences :@sentences, scene_graph_triplets: @scene_graph_triplets, updates: 1} UPDATE \
                { updates: OLD.updates + 1, description: @description, full_path: @file_name, scene_element: @scene_element, mdfs:  @mdf, start: @start\
                , stop: @stop, sentences: @sentences, scene_graph_triplets: @scene_graph_triplets} IN StoryLine \
                    RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
        bind_vars = {
                        'movie_id': movie_id,
                        'arango_id': arango_id,
                        'scene_element': scene_element,
                        'mdf': mdf, 
                        'start': start, 
                        'stop': stop,
                        'sentences': sentences,
                        'scene_graph_triplets': triplets,
                        'description': description,
                        'file_name': file_name
                        }
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        # for doc in cursor:
        #     doc=doc
        # return(doc['doc']['_id'])
 

    def create_story_line(self, fn, arango_id, movie_id, scene_element, start_frame, stop_frame, mdfs):        
        if (fn):
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            if video_file.is_file():
                #Simple MDF search - first, start and middle - to be enchanced 
                print("Scene: ", scene_element )
                cap = cv2.VideoCapture(fn)
                feature_mdfs = []
                for count, mdf in enumerate(mdfs):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mdf)
                    ret, frame_ = cap.read() # Read the frame
                    
                    
                    if not ret:
                        print("File not found")
                    else:
                        feature_t = self._calculate_images_features(frame_)
                        feature_mdfs.append(feature_t)                 
                feature_data = {'video_path': file_name,
                                'start_frame_id': start_frame,
                                'stop_frame_id': stop_frame,
                                'scene_id': scene_element,
                                #'frame_id': mdf,
                                'mdfs': mdfs,
                                'movie_id' : movie_id,
                                'arango_id' : arango_id
                                #'frame': frame_.tolist()
                                }
                self._add_image_features(feature_mdfs, feature_data)              
                cap.release()
                cv2.destroyAllWindows()
                cap.release()
                cv2.destroyAllWindows()
                    #print (start_frame)                     
        else:
            print("File doesn't exist: ", fn)


    def _calculate_images_features(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #image = Image.fromarray(image)
        image = self.preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features

    def _add_image_features(self, new_features_mdfs, feature_video_map):
        scene_graph_triplets = []
        scene_stories = []
        #print("FET. LEN: ", len(new_features_mdfs))
        for i, new_features in enumerate(new_features_mdfs):
            scene_graph_triplets_mdf = []
            scene_stories_mdf = []
            new_features /= new_features.norm(dim=-1, keepdim=True)
            vector = new_features.tolist()[0]
            search_scene_graph = self.scene_graph.search_vector(3, vector)
            for distance, data in search_scene_graph:
                if data['stage'] not in scene_graph_triplets_mdf:
                    if distance > 0.37:
                        scene_graph_triplets_mdf.append(data['stage'])
                        #print(scene_graph_triplets_mdf)
                        #stage_stories.append(data['sentence'])
            search_stories = self.pegasus_stories.search_vector(3, vector)
            for distance, data in search_stories:
                if data['stage'] not in scene_stories_mdf:
                    if distance > 0.39:
                        scene_stories_mdf.append(data['sentence'])
                        #print(scene_stories_mdf)
            scene_graph_triplets.append(scene_graph_triplets_mdf)
            scene_stories.append(scene_stories_mdf)
            #print("MDF: ", i)
            #print("Story: ", scene_stories)
            #print("Scene; ", scene_graph_triplets)
            #print("Then...............")
        #Insert Meta into Arango StoryLine Here
        self.insert_scene_to_storyline(feature_video_map['video_path'], feature_video_map['arango_id'], \
            feature_video_map['movie_id'],feature_video_map['scene_id'], \
            feature_video_map['mdfs'], "", feature_video_map['start_frame_id'], \
            feature_video_map['stop_frame_id'], scene_stories, scene_graph_triplets)
        


def main():
    story_line = STORY_LINE_API()
    all_movies = story_line.get_all_movies_meta()
    for movie in all_movies.values():
        print("Processing Movie: ", movie)
        for i, scene_element in enumerate(movie['scene_elements']):
            file_name = movie['full_path']
            movie_id = movie['movie_id']
            arango_id = movie['_id']
            mdfs = movie['mdfs'][i]
            start_frame = scene_element[0]
            stop_frame = scene_element[1]
            stage = i
            story_line.create_story_line(file_name, movie_id, arango_id, stage, start_frame, stop_frame, mdfs)
if __name__ == "__main__":
    main()

