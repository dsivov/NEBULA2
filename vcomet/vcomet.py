import json
import glob
from pathlib import Path
import os.path
import os

from tqdm.std import tqdm
from nebula_api.milvus_api import MilvusAPI
import torch
import cv2
from PIL import Image
import numpy as np
import heapq
import re
from nebula_api.nebula_enrichment_api import NRE_API

from nebula_api.mdmmt_api.mdmmt_api import MDMMT_API
import urllib.request
import itertools
import spacy

class VCOMET_KG:
    def __init__(self):
        self.milvus_events = MilvusAPI(
            'milvus', 'vcomet_mdmmt_embedded_event', 'nebula_visualcomet', 1536)
        self.milvus_places = MilvusAPI(
            'milvus', 'vcomet_mdmmt_embedded_place', 'nebula_visualcomet', 1536)
        self.milvus_actions = MilvusAPI(
            'milvus', 'vcomet_mdmmt_embedded_actions', 'nebula_visualcomet', 1536)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nre = NRE_API()
        self.db = self.nre.db
        self.gdb = self.nre.gdb
        self.mdmmt = MDMMT_API()
        self.temp_file = "/tmp/video_name.mp4" 
        self.en = spacy.load('en_core_web_sm')
        
    def download_video_file(self, movie):
        if os.path.exists('/tmp/video_file_dima.mp4'):
            os.remove('/tmp/video_file_dima.mp4')
        query = 'FOR doc IN Movies FILTER doc._id == "{}" RETURN doc'.format(movie)
        cursor = self.db.aql.execute(query)
        url_prefix = "http://ec2-3-120-189-231.eu-central-1.compute.amazonaws.com:7000/"
        url_link = ''
        for doc in cursor:
            url_link = url_prefix+doc['url_path']
            url_link = url_link.replace(".avi", ".mp4")   
            print(url_link)
            urllib.request.urlretrieve(url_link, self.temp_file) 
        video = cv2.VideoCapture(self.temp_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        return(fps, url_link)

    def get_stages(self, m):
        query_r = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}" RETURN doc'.format(
            m)
        cursor_r = self.db.aql.execute(query_r)
        stages = []
        for stage in cursor_r:
            stages.append(stage)
        return(stages)

    
    def get_actions_and_intents_for_place(self, event):
        vc_db = self.gdb.connect_db("nebula_visualcomet")
        query = 'FOR doc IN vcomet_kg FILTER doc.event == "{}" RETURN doc'.format(event)
        cursor = vc_db.aql.execute(query)
        for doc in cursor:
            print("         Proposed Intent...")
            print(doc['intent'])
            print("         Proposed actions")
            print(doc['after'])            
            print(doc['before'])            


    def get_places_and_events_for_scene(self, movie):
        stages = self.get_stages(movie)
        movie_candidates = []
        vectors = []
        for stage in stages:
            stage_candidates_events = []
            stage_candidates_places = []
            stage_candidates_actions = []
            print("Find candidates for scene")
            #input()
            path = ""            
            mdmmt_v, url_link = self.mdmmt_video_encode(stage['start'], stage['stop'], movie)
            vectors.append(mdmmt_v)
            if mdmmt_v is not None:
                vector = mdmmt_v.tolist()
                # print(vector)
                # input()
                similar_nodes = self.milvus_events.search_vector(40, vector)
                max_sim = similar_nodes[0][0]
                #print("Candidate Events in scene...")
                for node in similar_nodes:
                    if (max_sim - node[0]) > 0.05:
                        break
                    #max_sim = node[0]
                    stage_candidates_events.append([node[0], node[1]['sentence']])
                    #print('-----------> '+  str(node[1]['sentence']))
                
                similar_nodes = self.milvus_places.search_vector(40, vector)
                max_sim = similar_nodes[0][0]
                #print("Candidate Places of scene")
                for node in similar_nodes:
                    if (max_sim - node[0]) > 0.05:
                        break
                    #max_sim = node[0]
                    stage_candidates_places.append([node[0], node[1]['sentence']])
                    #print('-----------> '+  str(node[1]['sentence']))
                
                similar_nodes = self.milvus_actions.search_vector(40, vector)
                max_sim = similar_nodes[0][0]
                #print("Candidate Actions of scene")
                for node in similar_nodes:
                    if (max_sim - node[0]) > 0.05:
                        break
                    #max_sim = node[0]
                    stage_candidates_actions.append([node[0], node[1]['sentence']])
                    #print('-----------> ' + str(node[1]['sentence']))
            movie_candidates.append({
                                    'scene_element': stage['scene_element'],
                                    'start': stage['start'], 
                                    'stop': stage['stop'],
                                    'events': stage_candidates_events,
                                    'places': stage_candidates_places,
                                    'actions': stage_candidates_actions
                                                        })
        return(movie_candidates, url_link)

    def mdmmt_video_encode(self, start_f, stop_f, movie):
        import sys
        mdmmt = self.mdmmt
        path = self.temp_file
        fps, url_link = self.download_video_file(movie)
        t_start = start_f//fps
        t_end = stop_f//fps
        if t_start == t_end:
            t_start = t_start - 1
        print("Start/stop", t_start, " ", t_end)
        if ((t_end - t_start) >= 1) and (t_start >=0):
            vemb = mdmmt.encode_video(
                mdmmt.vggish_model,  # adio modality
                mdmmt.vmz_model,  # video modality
                mdmmt.clip_model,  # image modality
                mdmmt.model_vid,  # aggregator
                path, t_start, t_end)
            return(vemb, url_link)
        else:
            print("Stage too short")
            return(None, None)

    def get_playground_movies(self):
        return([
                "Movies/92354428",
                "Movies/92357362",
                "Movies/92356735",
                "Movies/92363515",
                "Movies/97689709",
                "Movies/92363218",
                "Movies/113659305",
                "Movies/114206816",
                "Movies/114206849",
                "Movies/114206892",
                "Movies/114206952",
                "Movies/114206999",
                "Movies/114207112",
                "Movies/114207139",
                "Movies/114207205",
                "Movies/114207240",
                "Movies/114207265",
                "Movies/114207324",
                "Movies/114207361",
                "Movies/114207398",
                "Movies/114207441",
                "Movies/114207474",
                "Movies/114207499",
                "Movies/114207550",
                "Movies/114207631",
                "Movies/114207668",
                "Movies/114207705",
                "Movies/114207740",
                "Movies/114207781",
                "Movies/114207810",
                "Movies/114207839",
                "Movies/114207908",
                "Movies/114207953",
                "Movies/114207984",
                "Movies/114208031",
                "Movies/114208064",
                "Movies/114208149",
                "Movies/114208196",
                "Movies/114208253",
                "Movies/114208338",
                "Movies/114208367",
                "Movies/114208576",
                "Movies/114208637",
                "Movies/114208744",
                "Movies/114208777",
                "Movies/114208820",
                "Movies/114206358",
                "Movies/114206264",
                "Movies/114206337",
                "Movies/114206397",
                "Movies/114206632",
                "Movies/114206724",
                "Movies/114206597",
                "Movies/114206691",
                "Movies/114206789",
                "Movies/114207184",
                "Movies/114206548"
                ])

    def insert_playgound_embeddings(self):
        vc_collection = self.db.collection("nebula_vcomet_lighthouse")
        movies = self.get_playground_movies()
        for movie in movies:
            stage_data, url_link = self.get_places_and_events_for_scene(movie)
            for s in stage_data:
                s['movie'] = movie
                s['url_link'] = url_link
                vc_collection.insert(s)
   
    def test_split(self, text):
        #import deplacy
       
        splits = []
        #text = 'man is holding onto a mackeral and tossing his head back laughing with somebody else'
        doc = self.en(text)
        #deplacy.render(doc)

        seen = set() # keep track of covered words

        chunks = []
        for sent in doc.sents:
            heads = [cc for cc in sent.root.children if cc.dep_ == 'conj']

            for head in heads:
                words = [ww for ww in head.subtree]
                for word in words:
                    seen.add(word)
                chunk = (' '.join([ww.text for ww in words]))
                chunks.append( (head.i, chunk) )

            unseen = [ww for ww in sent if ww not in seen]
            chunk = ' '.join([ww.text for ww in unseen])
            chunks.append( (sent.root.i, chunk) )

        chunks = sorted(chunks, key=lambda x: x[0])

        for ii, chunk in chunks:
            splits.append(chunk)
        return(splits)

    def get_text_img_score(self, text, img_emb):
        text_emb = self.encode_text(text)
        text_emb = text_emb / np.linalg.norm(text_emb)
        return np.sum((text_emb * img_emb))
    
    def get_top_k_from_proposed(self,top_k, proposed, embedding_array):
        proposed_map = {}
        candidates_text = []
        candidates_score = []
        for ev in tqdm(proposed):
                score = self.get_text_img_score(ev, embedding_array)
                proposed_map[score] = ev
        top_k = heapq.nlargest(top_k, proposed_map)
        for k in top_k:
            candidates_score.append(k)
            candidates_text.append(proposed_map[k])
            print (k, "->", proposed_map[k])
        return(candidates_score ,candidates_text)
    
  
def main():  
    kg = VCOMET_KG()
    movies = kg.get_playground_movies()
    for movie in movies:
        data = kg.nre.get_vcomet_data(movie)
        for scene_element in data:
            start_ = scene_element['start']
            stop_ = scene_element['stop']
            #vector, url = kg.mdmmt_video_encode(start_, stop_, movie)
            #if url:
            events = []
            proposed_events = []
            for event in scene_element['events']:
                print(event)
#zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)
if __name__ == "__main__":
    main()


