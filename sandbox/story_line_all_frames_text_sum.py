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
from torch._C import set_num_interop_threads
# from patchify import patchify
from nebula_api.milvus_api import MilvusAPI
from nebula_api.nebula_enrichment_api import NRE_API

import spacy
from gensim.parsing.preprocessing import remove_stopwords

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from benchmark.clip_benchmark import NebulaVideoEvaluation
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words



class STORY_LINE_SUM:
    def __init__(self):

        logging.basicConfig(format='%(asctime)s - %(message)s',
                            level=logging.INFO)

        self.clip_bench = NebulaVideoEvaluation()

        
        self.nlp = spacy.load('en_core_web_sm')
        #self.excluded_tags = {"NOUN", "VERB", "ADJ", "ADV", "ADP", "PROPN"}
        self.excluded_tags = {"VERB", "ADV", "ADP", "PROPN"}
        # Choose device and load the chosen model
        self.collection_name = 'nebula_clip'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("RN50x4", self.device)
        #self.model_test, self.preprocess_test = clip.load("RN101", self.device)
        self.scene_graph = MilvusAPI('milvus', 'scene_graph_visual_genome', 'nebula_dev', 640) 
        self.pegasus_stories =  MilvusAPI('milvus', 'pegasus', 'nebula_dev', 640)
        self.nre = NRE_API()
        self.db = self.nre.db
        print(self.db)
        
    def get_all_movies_meta(self):
        nebula_movies={}
        #query = 'FOR doc IN Movies FILTER doc.split == \'0\' AND doc.splits_total == \'30\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc.status != "complited" RETURN doc'
        query = 'FOR doc IN Movies FILTER doc._id == \'Movies/114208744\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/17342682\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/12911567\' RETURN doc'

        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/11723602\' RETURN doc'
        cursor = self.db.aql.execute(query)
        for data in cursor: 
            nebula_movies[data['_id']] = data
        return(nebula_movies) 

    def get_movie_expert_data(self, movie_id, scene_element):
        objects = []
        actions = []

        o_query = 'FOR doc IN Nodes FILTER doc.bboxes != null AND doc.arango_id == "{}" AND doc.class == "Object" RETURN doc'.format(
            movie_id)
        a_query = 'FOR doc IN Nodes FILTER doc.arango_id == "{}" AND doc.class == "Actions"  RETURN doc'.format(
            movie_id)

        cursor = self.db.aql.execute(o_query)
        for data in cursor:
            if data['description'] not in objects:
                objects.append(data['description'])
        cursor = self.db.aql.execute(a_query)
        for data in cursor:
            if data['description'] not in actions:
                actions.append(data['description'])
        return(objects,actions)

    def get_movie_tracker_data(self, movie_id, scene_element, frame):
        _objects = []
        bboxes = []
        o_query = 'FOR doc IN Nodes FILTER doc.bboxes != null AND doc.arango_id == \
            @arango_id AND doc.scene_element == @scene_element AND doc.class == "Object" RETURN doc'
        bind_vars = {
            'arango_id': movie_id,
            'scene_element': int(scene_element)
        }
        #print(o_query)
        cursor = self.db.aql.execute(o_query, bind_vars=bind_vars)
        for data in cursor:
            if str(frame) in data['bboxes']:
                if data['scores'][str(frame)] > 0.60:
                    _objects.append(data['description'])
                    bboxes.append(data['bboxes'][str(frame)])

        return(_objects, bboxes)


    def create_story_line(self, fn, arango_id, movie_id, scene_element, start_frame, stop_frame, mdfs):        
        if (fn):
            video_file = Path(fn)
            file_name = fn
            if video_file.is_file():
               
                cap = cv2.VideoCapture(fn)
                fps = cap.get(cv2.CAP_PROP_FPS)
                feature_mdfs = []
                feature_data = []
                for mdf in range(start_frame,stop_frame):
                   
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mdf)
                    ret, frame_ = cap.read() # Read the frame
                  
                    if not ret:
                        print("File not found")
                    else:
                        feature_t = self._calculate_images_features(frame_)
                       
                        feature_mdfs.append(feature_t)                 
                        feature_data.append({'video_path': file_name,
                                        'start_frame_id': start_frame,
                                        'stop_frame_id': stop_frame,
                                        'scene_id': scene_element,
                                        'frame_id': mdf,
                                        'mdfs': mdfs,
                                        'movie_id' : movie_id,
                                        'arango_id' : arango_id
                                        })
                self._add_image_features(feature_mdfs, feature_data, fps)              
                cap.release()
                cv2.destroyAllWindows()
                cap.release()
                cv2.destroyAllWindows()
                    #print (start_frame)                     
        else:
            print("File doesn't exist: ", fn)
        return(fps)

    def _calculate_images_features(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
        image = self.preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features

    def get_text_img_score(self, text, clip_bench, img_emb):
        text_emb = clip_bench.encode_text(text)
        text_emb = text_emb / np.linalg.norm(text_emb)
        return np.sum((text_emb * img_emb))

    def _add_image_features(self, new_features_mdfs, feature_video_map, fps):
    
        sentences = []
        
        for obj, new_features in zip(feature_video_map, new_features_mdfs):
            summs = []
            new_features /= new_features.norm(dim=-1, keepdim=True)
            vector = new_features.tolist()[0]
            search_scene_graph = self.scene_graph.search_vector(200, vector)       
            for distance, data in search_scene_graph:             
                if distance > 0.40:      
                    if (data['sentence'] + ".") not in sentences:
                        sentences.append(data['sentence'] + ".")
                
            search_stories = self.pegasus_stories.search_vector(200, vector)
            for distance, data in search_stories:
                if distance > 0.40:    
                    if data['sentence'] not in sentences:
                        sentences.append(data['sentence'])
                     
        
        start = int(feature_video_map[0]['start_frame_id']) / fps
        stop = int(feature_video_map[0]['stop_frame_id']) / fps
        thresholds = [0.8]
        movie_name = feature_video_map[0]['video_path']
        embedding_list, boundaries = self.clip_bench.create_clip_representation(movie_name,
                                                                                start_time=start, end_time=stop, thresholds=thresholds, method='average')
        print(start, " ", stop)
        print(feature_video_map[0]['video_path'])

        story = ' '.join(sentences[:1000])
        
        for k in range(embedding_list[0].shape[0]):
            emb = embedding_list[0][k, :]

        parser = PlaintextParser.from_string(
            story, Tokenizer('english'))
        stemmer = Stemmer('english')
        stop_words = get_stop_words('english')
        
        lsa_summarizer = LsaSummarizer(stemmer)
        lsa_summarizer.stop_words = stop_words
        lsa_summary = lsa_summarizer(parser.document, sentences_count=10)
        for sentence in lsa_summary:
            if sentence not in summs:
                summs.append(sentence)

        kl_summarizer = KLSummarizer(stemmer)
        kl_summarizer.stop_words = stop_words
        kl_summary = kl_summarizer(parser.document, sentences_count=10)
        
        for sentence in kl_summary:
            if sentence not in summs:
                summs.append(sentence)
        lex_rank_summarizer = LexRankSummarizer(stemmer)
        lex_rank_summarizer.stop_words = stop_words
        lexrank_summary = lex_rank_summarizer(parser.document, sentences_count=10)
        
        for sentence in lexrank_summary:
            if sentence not in summs:
                summs.append(sentence)       
        luhn_summarizer = LuhnSummarizer()
        luhn_summary = luhn_summarizer(parser.document, sentences_count=20)
        
        for sentence in luhn_summary:
            if sentence not in summs:
                summs.append(sentence)
        for sum_ in summs:
            score = self.get_text_img_score(str(sum_), self.clip_bench, emb)
            if score > 0.372:
                print(sum_, "->", score)

def main():
    story_line = STORY_LINE_SUM()
    all_movies = story_line.get_all_movies_meta()
    #client = StanfordOpenIE(properties=properties)
    client = ""
    for movie in all_movies.values():
        print("Processing Movie: ", movie['_id'], "->", movie['movie_name'])
        for i, scene_element in enumerate(movie['scene_elements']):
            print("Scene Element: ", i)
            file_name = movie['full_path']
            movie_id = movie['movie_id']
            arango_id = movie['_id']
            mdfs = movie['mdfs'][i]
            start_frame = scene_element[0]
            stop_frame = scene_element[1]
            stage = i
            story_line.create_story_line(file_name, movie_id, arango_id, stage, start_frame, stop_frame, mdfs)
        obj, act = story_line.get_movie_expert_data(arango_id, stage)
        print("Tracker: ")
        print(obj)
        print("STEP ")
        print(act)
if __name__ == "__main__":
    main()

