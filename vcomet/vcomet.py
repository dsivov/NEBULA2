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


class VCOMET_KG:
    def __init__(self):
        self.milvus_events = MilvusAPI(
            'milvus', 'vcomet_mdmmt_embedded_event', 'nebula_visualcomet', 1536)
        self.milvus_places = MilvusAPI(
            'milvus', 'vcomet_mdmmt_embedded_place', 'nebula_visualcomet', 1536)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nre = NRE_API()
        self.db = self.nre.db
        self.gdb = self.nre.gdb
        self.mdmmt = MDMMT_API()
        self.temp_file = "/tmp/video_name.mp4" 
        
    def download_video_file(self, movie):
        if os.path.exists('/tmp/video_file.mp4'):
            os.remove('/tmp/video_file.mp4')
        query = 'FOR doc IN Movies FILTER doc._id == "{}" RETURN doc'.format(movie)
        cursor = self.db.aql.execute(query)
        url_prefix = "http://ec2-52-57-66-3.eu-central-1.compute.amazonaws.com:7000/"
        for doc in cursor:
            url_link = url_prefix+doc['url_path']
            url_link = url_link.replace(".avi", ".mp4")   
            print(url_link)
            urllib.request.urlretrieve(url_link, self.temp_file) 
        video = cv2.VideoCapture(self.temp_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        return(fps)

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
        places = []
        vectors = []
        for stage in stages:
            print("Find candidates for scene")
            #input()
            path = ""            
            mdmmt_v = self.mdmmt_video_encode(stage['start'], stage['stop'], movie)
            vectors.append(mdmmt_v)
            if mdmmt_v is not None:
                vector = mdmmt_v.tolist()
                # print(vector)
                # input()
                similar_nodes = self.milvus_events.search_vector(5, vector)
                max_sim = similar_nodes[0][0]
                print("Events in scene...")
                for node in similar_nodes:
                    if (max_sim - node[0]) > 0.05:
                        break
                    max_sim = node[0]
                    places.append(node[1]['sentence'])
                    print(node)
                    #self.get_actions_and_intents_for_place(node[1]['sentence'][1])
                similar_nodes = self.milvus_places.search_vector(5, vector)
                max_sim = similar_nodes[0][0]
                print("Places of scene")
                for node in similar_nodes:
                    if (max_sim - node[0]) > 0.05:
                        break
                    max_sim = node[0]
                    places.append(node[1]['sentence'])
                    print(node)

    def mdmmt_video_encode(self, start_f, stop_f, movie):
        import sys
        mdmmt = self.mdmmt
        path = self.temp_file
        fps = self.download_video_file(movie)
        t_start = start_f//fps
        t_end = stop_f//fps
        if t_start == t_end:
            t_start = t_start - 1
        print("Start/stop", t_start, " ", t_end)
        if (t_end - t_start) >= 1:
            vemb = mdmmt.encode_video(
                mdmmt.vggish_model,  # adio modality
                mdmmt.vmz_model,  # video modality
                mdmmt.clip_model,  # image modality
                mdmmt.model_vid,  # aggregator
                path, t_start, t_end)
            return(vemb)
        else:
            print("Stage too short")
            return(None)

def main():
    kg = VCOMET_KG()
    
    #movie = 'Movies/92361646'
    #movie = 'Movies/114208744'
    #movie = 'Movies/92360929'
    movie = 'Movies/92363155'
    kg.get_places_and_events_for_scene(movie)
    #kg.download_video_file(movie)

#zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)
if __name__ == "__main__":
    main()


