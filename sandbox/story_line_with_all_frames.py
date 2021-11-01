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
# from patchify import patchify
from nebula_api.milvus_api import MilvusAPI
from nebula_api.nebula_enrichment_api import NRE_API

from openie import StanfordOpenIE
import spacy
from gensim.parsing.preprocessing import remove_stopwords
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sumy
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from benchmark.clip_benchmark import NebulaVideoEvaluation



class STORY_LINE_API:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            'gpt2')
        self.sum_model = GPT2LMHeadModel.from_pretrained(
            'gpt2')

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
                    # print("Frame: ", frame, " ", data['description'], "->", data['bboxes'][str(
                    #     frame)], " Score: ", data['scores'][str(frame)])
       
        return(_objects, bboxes)

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
 

    def create_story_line(self, fn, arango_id, movie_id, scene_element, start_frame, stop_frame, mdfs, client):        
        if (fn):
            video_file = Path(fn)
            file_name = fn
            if video_file.is_file():
                #Simple MDF search - first, start and middle - to be enchanced 
                #print("Scene: ", scene_element )
                cap = cv2.VideoCapture(fn)
                feature_mdfs = []
                feature_obj = []
                feature_bbox = []
                feature_data = []
                for mdf in range(start_frame,stop_frame):
                    #print("Processing frame ----------->", mdf)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mdf)
                    ret, frame_ = cap.read() # Read the frame
                    #cv2.imwrite('test/full_{}.png'.format(mdf), frame_)
                    objects, bboxes = self.get_movie_tracker_data(movie_id, scene_element, mdf)
                    #for o,b in zip(objects, bboxes):
                    # x,y,w,h = b
                    # crop = frame_[y:h, x:w]
                    #cv2.imwrite('test/{}_{}.png'.format(mdf, o), crop)

                    if not ret:
                        print("File not found")
                    else:
                        feature_t = self._calculate_images_features(frame_)
                        # feature_obj.append(o)
                        # feature_bbox.append(b)
                        feature_mdfs.append(feature_t)                 
                        feature_data.append({'video_path': file_name,
                                        'start_frame_id': start_frame,
                                        'stop_frame_id': stop_frame,
                                        'scene_id': scene_element,
                                        'frame_id': mdf,
                                        'mdfs': mdfs,
                                        'movie_id' : movie_id,
                                        'arango_id' : arango_id
                                        #'bboxes': b,
                                        #'objects': o
                                        #'frame': frame_.tolist()
                                        })
                self._add_image_features(feature_mdfs, feature_data, client)              
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

    def get_text_img_score(self, text, clip_bench, img_emb):
        text_emb = clip_bench.encode_text(text)
        text_emb = text_emb / np.linalg.norm(text_emb)
        return np.sum((text_emb * img_emb))

    def _add_image_features(self, new_features_mdfs, feature_video_map, client):
        from collections import Counter
        objects = []
        subjects = []
        relations = []
        triplets = []
        sentences = []

        #print("FET. LEN: ", len(new_features_mdfs))
        #print("Processing frame: ", feature_video_map['frame_id'])
        for obj, new_features in zip(feature_video_map, new_features_mdfs):
            scene_graph_triplets_mdf = []
            scene_stories_mdf = []
            new_features /= new_features.norm(dim=-1, keepdim=True)
            vector = new_features.tolist()[0]
            search_scene_graph = self.scene_graph.search_vector(200, vector)
            #print("Processing frame: ", obj['frame_id'], " Detected Object: ", obj['objects'])
            for distance, data in search_scene_graph:
                #if data['stage'] not in scene_graph_triplets_mdf:
                if distance > 0.38:
                        #scene_graph_triplets_mdf.append(data['stage'])
                    #print(data['sentence'], "-----> ", distance)
                    if data['sentence'] not in sentences:
                        sentences.append(data['sentence'] + ".")
                    # filtered_text = remove_stopwords(
                    #         data['sentence'].lower())
                    #print(filtered_text)
                    
                    # for triplet in client.annotate(str(filtered_text)):
                    #     #print("Source->TRIPLET: ",triple, "----->", distance)
                    #     #if triple['object'] not in objects: 
                    #     objects.append(triplet['object'])
                    #     #if triple['subject'] not in subjects:
                    #     subjects.append(triplet['subject'])
                    #     #if triple['relation'] not in relations:
                    #     ts = self.nlp(triplet['relation'])
                    #     for token in ts:
                    #         relations.append(token.lemma_)
                    #         triplets.append([triplet['subject'], token.lemma_, triplet['object'] ])
                        #stage_stories.append(data['sentence'])
            search_stories = self.pegasus_stories.search_vector(200, vector)
            for distance, data in search_stories:
                #print(data['sentence'], "->", distance)
                if data['stage'] not in scene_stories_mdf:
                    if distance > 0.38:
                        #print(data['sentence'], "-----> ", distance)
                        if data['sentence'] not in sentences:
                            sentences.append(data['sentence'])
                        filtered_text = remove_stopwords(
                            data['sentence'].lower())
        
                        #print(filtered_text)
                        # for triplet in client.annotate(str(filtered_text)):
                        #     #print("Source->PEGASUS: ",triple, "----->", distance)
                        #     objects.append(triplet['object'])
                        #     #if triple['subject'] not in subjects:
                        #     subjects.append(triplet['subject'])
                        #     #if triple['relation'] not in relations:
                        #     ts = self.nlp(triplet['relation'])
                        #     for token in ts:
                        #         relations.append(token.lemma_)
                        #         triplets.append(
                        #             [triplet['subject'], token.lemma_, triplet['object']])
        # print("Subjects: ") 
        # print(subjects)
        # print("Objects: ")
        # print(objects)
        # print("Possible relations: ")
        # print(relations)
        
        start = int(feature_video_map[0]['start_frame_id']) / 29
        stop = int(feature_video_map[0]['stop_frame_id']) / 29
        thresholds = [0.6, 0.7, 0.8]
        embedding_list, boundaries = self.clip_bench.create_clip_representation("1010_TITANIC_00_41_32_072-00_41_40_196.mp4",
                                                                                start_time=start, end_time=stop, thresholds=thresholds, method='average')
        print(start, " ", stop)
        subjects_map = Counter(subjects)
        objects_map = Counter(objects)
        relations_map = Counter(relations)
        subjects_map = dict(sorted(subjects_map.items(), key=lambda item: item[1], reverse=True))
        objects_map = dict(sorted(objects_map.items(), key=lambda item: item[1], reverse= True))
        relations_map = dict(sorted(relations_map.items(), key=lambda item: item[1], reverse=True))
        story = ' '.join(sentences)
        
        for k in range(embedding_list[0].shape[0]):
            emb = embedding_list[0][k, :]

        parser = PlaintextParser.from_string(
            story, Tokenizer('english'))
        lsa_summarizer = LsaSummarizer()
        lsa_summary = lsa_summarizer(parser.document, sentences_count=20)

        # Printing the summary
        for sentence in lsa_summary:
            score = self.get_text_img_score(str(sentence), self.clip_bench, emb)
            if score > 0.37:
                print(sentence, "->" , score)
           
        print("===================================================")
        kl_summarizer = KLSummarizer()
        kl_summary = kl_summarizer(parser.document, sentences_count=20)
        # Printing the summary
        for sentence in kl_summary:
            score = self.get_text_img_score(str(sentence), self.clip_bench, emb)
            if score > 0.37:
                print(sentence, "->", score)
        print("===================================================")
        lex_rank_summarizer = LexRankSummarizer()
        lexrank_summary = lex_rank_summarizer(parser.document, sentences_count=20)
        # Printing the summary
        for sentence in lexrank_summary:
            score = self.get_text_img_score(str(sentence), self.clip_bench, emb)
            if score > 0.37:
                print(sentence, "->", score)
        print("===================================================")
        luhn_summarizer = LuhnSummarizer()
        luhn_summary = luhn_summarizer(parser.document, sentences_count=20)
        # Printing the summary
        for sentence in luhn_summary:
            score = self.get_text_img_score(str(sentence), self.clip_bench, emb)
            if score > 0.37:
                print(sentence, "->", score)

        # inputs = self.tokenizer.batch_encode_plus(
        #     [story], return_tensors='pt', max_length = 20, truncation=True)
        # summary_ids = self.sum_model.generate(inputs['input_ids'], early_stopping=True)
        # XLM_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # print(XLM_summary)
        # for triplet in triplets:
        #     if subjects_map[triplet[0]] > 20:
        #         print(triplet)
        # print("Proposed subjects:")
        #print(subjects_map)
        # print("Proposed objects:")
        #print(objects_map)
        # print("Proposed relations:")
        # print(relations_map)
        # for subj in subjects_map:
        #     if subjects_map[subj] > 28:
        #         for rel in relations_map:
        #             if relations_map[rel] > 28:
        #                 for obj in objects_map:
        #                      if objects_map[obj] > 28:
        #                         print(subj+"--->"+rel+"--->"+obj)
        

        


def main():
    properties = {
        'openie.affinity_probability_cap': 1 / 3,
        'resolve_coref': True,
    }
    story_line = STORY_LINE_API()
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
            story_line.create_story_line(file_name, movie_id, arango_id, stage, start_frame, stop_frame, mdfs, client)
        obj, act = story_line.get_movie_expert_data(arango_id, stage)
        print("Tracker: ")
        print(obj)
        print("STEP ")
        print(act)
if __name__ == "__main__":
    main()

