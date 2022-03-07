from nebula_api.milvus_api import MilvusAPI
import torch
import heapq
#import cv2
from nebula_api.nebula_enrichment_api import NRE_API

#from nebula_api.mdmmt_api.mdmmt_api import MDMMT_API
from experts.common.RemoteAPIUtility import RemoteAPIUtility
from nebula_api.vlmapi import VLM_API


class VCOMET_PLACES:
    def __init__(self):
        self.milvus_places = MilvusAPI(
            'milvus', 'vcomet_vit_embedded_place', 'nebula_visualcomet', 768)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nre = NRE_API()
        self.db = self.nre.db
        self.places_collection = self.db.collection("nebula_vcomet_places_playground")
        #self.gdb = self.nre.gdb
        self.clipmodel = VLM_API(model_name='clip_vit')
        self.mdmmtmodel = VLM_API()
    
    def get_playground_movies(self):
        return(['Movies/114206816', 'Movies/114206849', 'Movies/114206892', 'Movies/114206952', 'Movies/114206999', 'Movies/114207139', 'Movies/114207205', 'Movies/114207240', 'Movies/114207265', 'Movies/114207324', 'Movies/114207361', 'Movies/114207398', 'Movies/114207441', 'Movies/114207474', 'Movies/114207499', 'Movies/114207550', 'Movies/114207668', 'Movies/114207740', 'Movies/114207781', 'Movies/114207810', 'Movies/114207839', 'Movies/114207908', 'Movies/114207953', 'Movies/114207984', 'Movies/114208064', 'Movies/114208149', 'Movies/114208196', 'Movies/114208338', 'Movies/114208367', 'Movies/114208576', 'Movies/114208637', 'Movies/114208744', 'Movies/114208777', 'Movies/114208820', 'Movies/114206358', 'Movies/114206264', 'Movies/114206337', 'Movies/114206397', 'Movies/114206632', 'Movies/114206597', 'Movies/114206691', 'Movies/114206789', 'Movies/114207184', 'Movies/114206548'])
        
    def get_playground_movies_and_scenes(self):
        return([('Movies/114206952', 1),
                ('Movies/114207324', 0),
                ('Movies/114207908', 0),
                ('Movies/114208149', 2),
                ('Movies/114208196', 1),
                ('Movies/114208338', 0),
                ('Movies/114208576', 0),
                ('Movies/114208744', 0),
                ('Movies/114208744', 2),
                ('Movies/114206337', 0),
                ('Movies/114206337', 1),
                ('Movies/114206548', 0),
                ('Movies/114206548', 1)])

    def get_places_for_scene(self, movie, stage):
        movie_candidates = []
        vectors = []
        stage_candidates_events = []
        stage_candidates_places = []
        stage_candidates_actions = []
        candidates_places = []
        scored_places = {}
        top_places = {}
        #print("Find candidates for scene")
        #input()
        path = ""  
        url_prefix = "http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:7000/"
        url = self.nre.get_movie_url(movie)          
        clip_v = self.clipmodel.encode_video(movie, stage, class_name='clip_vit_f' )
        mdmmt_v = self.mdmmtmodel.encode_video(movie, stage)

        vectors.append(clip_v.tolist()[0])
        if clip_v is not None:
            clip_v = clip_v.tolist()[0]
            similar_nodes = self.milvus_places.search_vector(100, clip_v)
            max_sim = similar_nodes[0][0]
            #print("Candidate Places of scene")
            for node in similar_nodes:
                stage_candidates_places.append([node[0], node[1]['sentence']])
                candidates_places.append(node[1]['sentence'])
            mdmmvtext_v = self.mdmmtmodel.encode_text(candidates_places)
            scores = torch.matmul(mdmmvtext_v, mdmmt_v)
            for txt, score in zip(candidates_places, scores):
                scored_places[score.tolist()]= txt
            top_k = heapq.nlargest(8, scored_places)
            for k in top_k:
                top_places[k] = scored_places[k]
                #print('-----------> '+  str(node[1]['sentence']))
            
            # similar_nodes = self.milvus_actions.search_vector(40, vector)
            # max_sim = similar_nodes[0][0]
            # #print("Candidate Actions of scene")
            # for node in similar_nodes:
            #     if (max_sim - node[0]) > 0.05:
            #         break
            #     #max_sim = node[0]
            #     stage_candidates_actions.append([node[0], node[1]['sentence']])
            #     #print('-----------> ' + str(node[1]['sentence']))
        movie_candidates.append({
                                'movie_id': movie,
                                'scene_element': stage,
                                'start': '', 
                                'stop': '',
                                #'events': stage_candidates_events,
                                'places': top_places,
                                #'actions': stage_candidates_actions,
                                'url_link': url_prefix + url
                                                    })
        return(movie_candidates)

def main():
    kg = VCOMET_PLACES()
    movies = kg.get_playground_movies()
    for movie in movies:
        scene_elements = kg.nre.get_stages(movie)
        for scene_element in scene_elements:
            places = kg.get_places_for_scene(movie, scene_element['scene_element'])
            print(places)
            input()
        #kg.places_collection.insert(places)
    
if __name__ == "__main__":
    main()