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
from experts.common.RemoteAPIUtility import RemoteAPIUtility


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
        self.temp_file = "/home/ilan/git/NEBULA2-latest/NEBULA2/video_file.mp4" #"/tmp/video_file.mp4" 
        # self.en = spacy.load('en_core_web_sm')
        
    def download_video_file(self, movie):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        query = 'FOR doc IN Movies FILTER doc._id == "{}" RETURN doc'.format(movie)
        cursor = self.db.aql.execute(query)
        url_prefix = "http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:7000/"
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

    # Bug in Movies/114206816 in stage 1
    def get_places_and_events_for_scene(self, movie):
        stages = self.get_stages(movie)
        movie_candidates = []
        url_link = ''
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
                                    'actions': stage_candidates_actions,
                                    'url_link': url_link
                                                        })
        return(movie_candidates, url_link)


    # def get_places_and_events_for_scene_modified(self, t_start, t_end, path):  

    #     mdmmt_vs = self.mdmmt_video_encode_modified(t_start, t_end, path)
    #     for mdmmt_v in mdmmt_vs:
    #         if mdmmt_v is not None:

    #             vector = mdmmt_v.tolist()
    #             # print(vector)
    #             # input()
    #             similar_nodes = self.milvus_events.search_vector(3, vector)
    #             max_sim = similar_nodes[0][0]
    #             #print("Candidate Events in scene...")
    #             print('')
    #             for node in similar_nodes:
    #                 if (max_sim - node[0]) > 0.05:
    #                     break
    #                 #max_sim = node[0]
    #                 print('Events: -----------> '+  str(node[1]['sentence']))
                
    #             similar_nodes = self.milvus_places.search_vector(3, vector)
    #             max_sim = similar_nodes[0][0]
    #             #print("Candidate Places of scene")
    #             for node in similar_nodes:
    #                 if (max_sim - node[0]) > 0.05:
    #                     break
    #                 #max_sim = node[0]
    #                 print(' Places: -----------> '+  str(node[1]['sentence']))
                
    #             similar_nodes = self.milvus_actions.search_vector(3, vector)
    #             max_sim = similar_nodes[0][0]
    #             #print("Candidate Actions of scene")
    #             for node in similar_nodes:
    #                 if (max_sim - node[0]) > 0.05:
    #                     break
    #                 #max_sim = node[0]
    #                 print(' Actions: -----------> ' + str(node[1]['sentence']))
    #             print('\n')
    
    def mdmmt_video_encode(self, start_f, stop_f, movie):
        import sys
        mdmmt = self.mdmmt
        path = self.temp_file
        fps, url_link = self.download_video_file(movie)
        t_start = start_f // fps
        t_end = stop_f // fps
        if t_end < 1:
            t_end = stop_f / fps
        if t_start == t_end and t_start >= 1:
            t_start = t_start - 1
        
        length = t_end - t_start
        print("---------------")
        print(f"Movie ID: {movie}")#
        print("Start/stop", t_start, " ", t_end)
        
        # if ((t_end - t_start) >= 1) and (t_start >=0):
        vemb = mdmmt.encode_video(
            mdmmt.vggish_model,  # audio modality
            mdmmt.vmz_model,  # video modality
            mdmmt.clip_model,  # image modality
            mdmmt.model_vid,  # aggregator
            path, t_start, t_end, fps=fps, encode_type='mean')
        return(vemb, url_link)
        # else:
        #     print("Stage too short")
        #     return(None, None)


    # def mdmmt_video_encode_modified(self, t_start, t_end, path):

    #     mdmmt = self.mdmmt

    #     vemb = mdmmt.encode_video(
    #         mdmmt.vggish_model,  # audio modality
    #         mdmmt.vmz_model,  # video modality
    #         mdmmt.clip_model,  # image modality
    #         mdmmt.model_vid,  # aggregator
    #         path, t_start, t_end)
    #     return vemb
  


    def get_playground_movies(self):
        return(['Movies/114206816', 'Movies/114206849', 'Movies/114206892', 'Movies/114206952', 'Movies/114206999', 'Movies/114207139', 'Movies/114207205', 'Movies/114207240', 'Movies/114207265', 'Movies/114207324', 'Movies/114207361', 'Movies/114207398', 'Movies/114207441', 'Movies/114207474', 'Movies/114207499', 'Movies/114207550', 'Movies/114207668', 'Movies/114207740', 'Movies/114207781', 'Movies/114207810', 'Movies/114207839', 'Movies/114207908', 'Movies/114207953', 'Movies/114207984', 'Movies/114208064', 'Movies/114208149', 'Movies/114208196', 'Movies/114208338', 'Movies/114208367', 'Movies/114208576', 'Movies/114208637', 'Movies/114208744', 'Movies/114208777', 'Movies/114208820', 'Movies/114206358', 'Movies/114206264', 'Movies/114206337', 'Movies/114206397', 'Movies/114206632', 'Movies/114206597', 'Movies/114206691', 'Movies/114206789', 'Movies/114207184', 'Movies/114206548'])
        # return(['Movies/114206816', 'Movies/114206849', 'Movies/114206892', 'Movies/114206952'])

    def insert_playgound_embeddings(self):
        vc_collection = self.db.collection("nebula_vcomet_lighthouse_lsmdc_mean_v01")
        movies = self.get_playground_movies()
        for movie in movies:
            stage_data, url_link = self.get_places_and_events_for_scene(movie)
            for s in stage_data:
                s['movie'] = movie
                # s['url_link'] = url_link
                if s['url_link'] is not None:
                    vc_collection.insert(s)

    '''
    This functions adds the 44 movie clips to nebula_development,
    In the document `nebula_mdmmt_vector_playground`.
    Every row will consist of `movie_id`, `scene_element`, `embedding`.
    '''
    def insert_playgound_by_mid_embeddings(self):
        vc_collection = self.db.collection("nebula_mdmmt_vector_playground_v01")
        movies = self.get_playground_movies()
        for movie in movies:
            row_dict = {
                'movie_id': movie,
                'scene_element': -1,
                'embedding': None
            }
            stage_data, url_link = self.get_places_and_events_for_scene(movie)
            for s in stage_data:
                row_dict['scene_element'] = s['scene_element']
                mdmmt_v, _ = self.mdmmt_video_encode(s['start'], s['stop'], movie)
                print(s['start'], s['stop'])
                if mdmmt_v is not None:
                    row_dict['embedding'] = mdmmt_v.cpu().numpy().tolist()
                    # vc_collection.insert([row_dict])
        

    
    def print_movie_by_id(self,
       movie_id,
       num_of_items = 1,
       db_orig = 'nebula_vcomet_lighthouse_lsmdc',
       db_max = 'nebula_vcomet_lighthouse_lsmdc_max',
       db_mean = 'nebula_vcomet_lighthouse_lsmdc_mean'
       ):

        # vc_collection = self.db.collection("nebula_vcomet_lighthouse_lsmdc_mean")
        query_orig = f'FOR doc IN {db_orig} RETURN doc'
        query_max = f'FOR doc IN {db_max} RETURN doc'
        query_mean = f'FOR doc IN {db_mean} RETURN doc'
        quries = [query_orig, query_max, query_mean]
        quries_names = [db_orig, db_max, db_mean]
        num_of_scenes = -1
        cursor = self.db.aql.execute(query_orig)
        for data in cursor:
                if data['movie'] == movie_id:
                    num_of_scenes += 1
        if num_of_scenes == -1:
            print(f"ERROR: {movie_id} not found!")
            return
        
        for cur_scene in range(num_of_scenes + 1):
            for idx, query in enumerate(quries):
                cursor = self.db.aql.execute(query)
                found = False
                for data in cursor:
                    
                    if data['movie'] == movie_id and data["scene_element"] == cur_scene:
                        print('########################################')
                        print(f"Current document: {quries_names[idx]}")
                        print(f'Movie: {data["movie"]}')
                        print(f'Movie URL: {data["url_link"]}')
                        print(f'Scene element: {data["scene_element"]}')
        
                        print("Movie events: ")
                        for row in range(num_of_items - 1):
                            if row < len(data["events"]):
                                print(data["events"][row])
                        
                        print("Movie places: ")
                        for row in range(num_of_items):
                            if row < len(data["places"]):
                                print(data["places"][row])

                        print("Movie actions: ")
                        for row in range(num_of_items):
                            if row < len(data["actions"]):
                                print(data["actions"][row]) 
                    

        


    def get_playgound_embeddings(self):
        movies = self.get_playground_movies()
        movies_dict = {}
        for movie in movies:
            stage_data, url_link = self.get_places_and_events_for_scene(movie)
            movies_dict[movie] = []
            print("\n #############################################")
            print(movie + ", " + url_link)
            print(url_link)
            print("\n -----------------------------")
            print("Number of scene elements: {}".format(len(stage_data)))
            for s in stage_data:
                movies_dict[movie].append(())
                print(s['scene_element'])
                print(s['start'])
                print(s['stop'])

   
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


    def download_and_get_minfo(self, mid):
        # Download the video locally
        fps, url_link = self.download_video_file(mid)
        api = RemoteAPIUtility()
        movie_info = api.get_movie_info(mid)
        return movie_info

  
def main():  

    #bad = [movie = "Movies/114206816", 114207361 - wierd,]
    kg = VCOMET_KG()
    kg.insert_playgound_by_mid_embeddings()
    # kg.get_playgound_embeddings()
    # a=0
    # kg.insert_playgound_by_mid_embeddings()
    # kg.get_playgound_embeddings()
    # movie_info = kg.download_and_get_minfo("Movies/114206816")
    # kg.mdmmt.encode_video_max()
    # print(movie_info)
    # kg.insert_playgound_embeddings()
    # kg.insert_playgound_by_mid_embeddings()
    # a=0
    # kg.print_movie_by_id(
    #    movie_id = 'Movies/114206816',
    #    num_of_items = 3
    # )
    # test_clips = ['1040_The_Ugly_Truth_00.51.36.680-00.51.38.232', '1031_Quantum_of_Solace_00.52.35.159-00.52.37.144', '1008_Spider-Man2_00.37.21.781-00.37.25.010', '1031_Quantum_of_Solace_00.39.09.510-00.39.14.286', '1006_Slumdog_Millionaire_01.50.17.425-01.50.20.715', '1052_Harry_Potter_and_the_order_of_phoenix_00.14.58.068-00.15.00.314', '0028_The_Crying_Game_00.53.53.876-00.53.55.522', '0004_Charade_00.08.11.578-00.08.11.963', '1009_Spider-Man3_00.42.59.783-00.43.01.608', '0033_Amadeus_00.16.03.665-00.16.08.486', '1038_The_Great_Gatsby_00.57.29.452-00.57.31.831', '0017_Pianist_00.34.12.556-00.34.15.845', '1049_Harry_Potter_and_the_chamber_of_secrets_00.19.45.874-00.19.49.051', '0021_Rear_Window_00.27.37.810-00.27.39.810', '1038_The_Great_Gatsby_00.56.48.259-00.56.50.126', '1047_Defiance_01.31.42.519-01.31.46.181', '0009_Forrest_Gump_00.43.57.991-00.44.00.160', '1023_Horrible_Bosses_01.24.36.860-01.24.38.899', '1029_Pride_And_Prejudice_Disk_One_02.28.27.683-02.28.31.276', '1055_Marley_and_me_00.03.35.270-00.03.36.189', '0029_The_Graduate_00.04.51.868-00.04.53.081', '0025_THE_LORD_OF_THE_RINGS_THE_RETURN_OF_THE_KING_02.55.46.515-02.55.51.946', '1005_Signs_00.10.56.732-00.11.00.017', '0011_Gandhi_01.05.17.564-01.05.18.429', '0014_Ist_das_Leben_nicht_schoen_01.14.53.158-01.14.53.866', '1035_The_Adjustment_Bureau_00.01.40.825-00.01.46.814', '1024_Identity_Thief_00.01.43.655-00.01.47.807', '1035_The_Adjustment_Bureau_00.16.58.881-00.17.05.736', '1047_Defiance_00.52.07.009-00.52.07.978', '0014_Ist_das_Leben_nicht_schoen_00.01.45.481-00.02.06.641', '1043_Vantage_Point_00.38.48.473-00.38.52.599', '1010_TITANIC_00.41.32.072-00.41.40.196', '1047_Defiance_01.08.28.259-01.08.29.433', '0030_The_Hustler_01.21.48.576-01.21.52.523', '0001_American_Beauty_00.21.38.688-00.21.39.904', '1005_Signs_00.14.35.680-00.14.40.450', '1015_27_Dresses_00.38.02.757-00.38.08.213', '0004_Charade_00.27.44.212-00.27.44.742', '0002_As_Good_As_It_Gets_00.06.32.767-00.06.33.455', '0027_The_Big_Lebowski_01.46.45.804-01.46.49.607', '1034_Super_8_01.42.41.370-01.42.45.709', '0026_The_Big_Fish_00.17.37.555-00.17.42.606', '1052_Harry_Potter_and_the_order_of_phoenix_02.00.04.103-02.00.06.616', '0001_American_Beauty_01.45.48.324-01.46.01.008']
    # for i, test_clip in enumerate(test_clips):
    #     test_clips[i] = '_'.join(test_clips[i].rsplit('.', 6))
    # kg = VCOMET_KG()
    # clips = []
    # movies_in_lsmdc = []
    # movies = kg.get_playground_movies()
    # for movie in movies:
    #     data = kg.nre.get_vcomet_data(movie)  
    #     cur_clip = data[0]['url_link'].split("/")[-1].replace(".mp4", "")
    #     if cur_clip in test_clips:
    #         clips.append(movie)
    #         print(cur_clip)

    #     # print("/n ############################################")    
    #     # [print(i['places']) for i in data]
    #     # [print(i['url_link']) for i in data]
    #     # print('.'.join(data[0]['url_link'].split("/")[-1].replace(".mp4", "").rsplit('_', 6)))
    #     # clips.append('.'.join(data[0]['url_link'].split("/")[-1].replace(".mp4", "").rsplit('_', 6)))
    #     # print("/n ###########################################")
    # print(clips)
    # # kg.insert_playgound_embeddings()
    # print("Done.")
    

    # for test_clip in test_clips:
    #     test_clip_temp = ('_'.join(test_clips[0].rsplit('.', 6))
    #     if test_clip_temp in clips:
    #         movies_in_lsmdc()

    


    
#zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)
if __name__ == "__main__":
    main()


