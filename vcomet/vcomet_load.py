from pathlib import Path
import os.path
import os

from tqdm.std import tqdm
from nebula_api.milvus_api import MilvusAPI
import torch
#import cv2
from PIL import Image
import numpy as np
import heapq
import re
from nebula_api.nebula_enrichment_api import NRE_API

#from nebula_api.mdmmt_api.mdmmt_api import MDMMT_API
from experts.common.RemoteAPIUtility import RemoteAPIUtility
from nebula_api.vlmapi import VLM_API

class VCOMET_LOAD:
    def __init__(self):
        self.milvus_events = MilvusAPI(
            'milvus', 'vcomet_vit_embedded_event', 'nebula_visualcomet', 768)
        self.milvus_places = MilvusAPI(
            'milvus', 'vcomet_vit_embedded_place', 'nebula_visualcomet', 768)
        self.milvus_actions = MilvusAPI(
            'milvus', 'vcomet_vit_embedded_actions', 'nebula_visualcomet', 768)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nre = NRE_API()
        self.db = self.nre.db
        self.gdb = self.nre.gdb
        self.vlmodel = VLM_API(model_name='clip_vit')
        self.vc_db = self.gdb.connect_db("nebula_visualcomet")

    
    def load_vit_vcomet_place(self):
        query = 'FOR doc IN vcomet_kg RETURN DISTINCT doc.place'
        cursor = self.vc_db.aql.execute(query)
        for vc in cursor:
            if len(vc.split()) < 9:  
                vector = self.vlmodel.encode_text(vc, class_name='clip_vit')
                print(len(vector.tolist()[0]))
                meta = {
                            'filename': 'none',
                            'movie_id':'none',
                            'nebula_movie_id': 'none',
                            'stage': 'none',
                            'frame_number': 'none',
                            'sentence': vc,
                        }
                self.milvus_places.insert_vectors([vector.tolist()[0]], [meta])
                #print(meta)
                #input()

def main():
    kg = VCOMET_LOAD()
    kg.load_vit_vcomet_place()
    
if __name__ == "__main__":
    main()