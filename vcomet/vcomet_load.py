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
            'milvus', 'vcomet_vit_embedded_event', 'nebula_visualcomet', 1536)
        self.milvus_places = MilvusAPI(
            'milvus', 'vcomet_vit_embedded_place', 'nebula_visualcomet', 1536)
        self.milvus_actions = MilvusAPI(
            'milvus', 'vcomet_vit_embedded_actions', 'nebula_visualcomet', 1536)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nre = NRE_API()
        self.db = self.nre.db
        self.gdb = self.nre.gdb
        self.vlmodel = VLM_API(model_name='clip_vit')
    
    def load_database(self):
        vector = self.vlmodel.encode_text("test", class_name='clip_vit')
        print(vector.size())

def main():
    kg = VCOMET_LOAD()
    kg.load_database()
    
if __name__ == "__main__":
    main()