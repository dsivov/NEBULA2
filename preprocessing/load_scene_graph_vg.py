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
from csv import reader
import json


data_dir = 'split/'
milvus_vg = MilvusAPI('milvus','scene_graph_visual_genome', 'nebula_dev', 640)
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
    
    data_dir = 'vgsplit/'
    #for __file_ in os.listdir(data_dir):
    
    sentences = []
    relations = []
    #print(data_dir + __file_)
    #my_file_name = data_dir + __file_
    my_file_name = "relationships_multy.json"
    my_file = open(my_file_name, "r")
    relation_dic = json.load(my_file)
    count = 0
    for i, row in enumerate(relation_dic):
       #print(i, row)
        
        for rels in row['relationships']:
            relation = []
            if 'name' in rels['subject'] and 'name' in rels['object']:
                subject_= rels['subject']['name'].lower()
                predicate = rels['predicate'].lower()
                object_ = rels['object']['name'].lower()
                sentence = subject_ + " " + predicate + " " + object_
                relation.append(subject_)
                relation.append(predicate)
                relation.append(object_)
                #if len(object_) > 60 or len(subject_) > 60 or len(predicate) > 60: 
                #print(sentence)
                #print(relation)
                if sentence not in sentences:
                    sentences.append(sentence)
                    relations.append(relation)
                    text_inputs = torch.cat([clip.tokenize(sentence)]).to(device)
                    with torch.no_grad():
                        text_features = model.encode_text(text_inputs)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        vector = text_features.tolist()[0]
                        #embeddings.append(text_features.tolist()[0])
                        meta = {
                            'filename': 'none',
                            'movie_id':'none',
                            'nebula_movie_id': 'none',
                            'stage': relation,
                            'frame_number': text_features.tolist()[0],
                            'sentence': sentence,
                        }
                        milvus_vg.insert_vectors([vector], [meta])
                    count = count + 1
            print(count)
    print(len(sentences))
    print(len(relations))
    #for i,j in zip()
       #print(i, " ", row['subject']['name'], " ", row['predicate'], " ", row['object']['name'] )
        # all_sentences = my_file.read().splitlines()
        # for sent in all_sentences:
        #     rels = parser(sent)
        #     #print(rels,"   " ,sent)
        #     for relation in rels:
        #         sentence = " ".join(relation)
        #         if sentence not in relations:
        #             text_inputs = torch.cat([clip.tokenize(sentence)]).to(device)
        #             with torch.no_grad():
        #                 text_features = model.encode_text(text_inputs)
        #                 text_features /= text_features.norm(dim=-1, keepdim=True)
        #                 embeddings.append(text_features.tolist()[0])
        #                 meta = {
        #                     'filename': 'none',
        #                     'movie_id':'none',
        #                     'nebula_movie_id': 'none',
        #                     'stage': relation,
        #                     'frame_number': text_features.tolist()[0],
        #                     'sentence': sentence,
        #                 }
        #                 metadata.append(meta)    
        #                 relations.append(sentence)

        # milvus_sg.insert_vectors(embeddings, metadata)
        
if __name__ == '__main__':
    main()

