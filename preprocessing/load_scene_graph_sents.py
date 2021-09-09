#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : demo.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 08/22/2018
#
# This file is part of SceneGraphParser.
# Distributed under terms of the MIT license.
# https://github.com/vacancy/SceneGraphParser

"""
A small demo for the scene graph parser.
"""
from gensim.parsing.preprocessing import remove_stopwords
import sng_parser
#from pprint import pprint
import os
#from arango.exceptions import IndexListError
from milvus_api.milvus_api import MilvusAPI
import os
import clip
import torch
data_dir = 'split/'
milvus_desc = MilvusAPI('milvus', 'descriptions', 'nebula_dev', 640)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50x4", device)


def main():
    
    data_dir = 'split/'
    for __file_ in os.listdir(data_dir):
        relations = []
        metadata = []
        embeddings = []
        print(data_dir + __file_)
        my_file_name = data_dir + __file_
        my_file = open(my_file_name, "r")
        all_sentences = my_file.read().splitlines()
        
 
        for sentence in all_sentences:       
            
            filtered_sentence = remove_stopwords(sentence)

            print(filtered_sentence)

            text_inputs = torch.cat([clip.tokenize(sentence)]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                print(len(text_features.tolist()[0]))
                embeddings.append(text_features.tolist()[0])
                meta = {
                    'filename': 'none',
                    'movie_id':'none',
                    'nebula_movie_id': 'none',
                    'stage': "",
                    'frame_number': text_features.tolist()[0],
                    'sentence': sentence,
                }
                metadata.append(meta)    
                relations.append(sentence)

        milvus_desc.insert_vectors(embeddings, metadata)
        
if __name__ == '__main__':
    main()

