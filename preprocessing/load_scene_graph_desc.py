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

import sng_parser
#from pprint import pprint
import os
#from arango.exceptions import IndexListError
from milvus_api.milvus_api import MilvusAPI
import os
import clip
import torch
data_dir = 'split/'
milvus_sg = MilvusAPI('milvus','scene_graph_triplets', 'nebula_dev', 640)
#milvus_desc = MilvusAPI('descriptions', 'nebula_dev', 512)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50x4", device)


def parser(sentence):
    #print('Sentence:', sentence)
    # Here we just use the default parser.
    #sng_parser.tprint(sng_parser.parse(sentence), show_entities=False)
    graph = sng_parser.parse(sentence)
    entities = graph['entities']
    relations_data = [
        [
            entities[rel['subject']]['head'].lower(),
            rel['relation'].lower(),
            entities[rel['object']]['head'].lower()
        ]
        for rel in graph['relations']
    ]
    return(relations_data)
    # print(graph)
    # print()


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
        for sent in all_sentences:
            rels = parser(sent)
            #print(rels,"   " ,sent)
            for relation in rels:
                sentence = " ".join(relation)
                if sentence not in relations:
                    text_inputs = torch.cat([clip.tokenize(sentence)]).to(device)
                    with torch.no_grad():
                        text_features = model.encode_text(text_inputs)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        embeddings.append(text_features.tolist()[0])
                        meta = {
                            'filename': 'none',
                            'movie_id':'none',
                            'nebula_movie_id': 'none',
                            'stage': relation,
                            'frame_number': text_features.tolist()[0],
                            'sentence': sentence,
                        }
                        metadata.append(meta)    
                        relations.append(sentence)

        milvus_sg.insert_vectors(embeddings, metadata)
        
if __name__ == '__main__':
    main()

