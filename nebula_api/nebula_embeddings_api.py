
from simpleneighbors import SimpleNeighbors
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import tokenize
from nebula_api.cfg import Cfg
from nebula_api.databaseconnect import DatabaseConnector
from milvus import Milvus, IndexType, MetricType, Status
#from metric_learn import NCA, LFDA, LMNN
from sentence_transformers import SentenceTransformer, util
from nebula_api.milvus_api import MilvusAPI
import numpy as np
import pandas as pd
import copy
import clip
import torch

from nebula_api.graph_encoder import GraphEncoder
from nebula_api.nebula_enrichment_api import NRE_API


class EmbeddingsLoader(): 
    
    def inference(self, _query):
        similars = {}
        print(list(tokenize(_query)))
        vector = self.encoder.infer_vector(list(tokenize(_query)))
        sims = self.index.nearest(vector, n=10)
        for sim in sims:    
            dist = 1
            similars[sim] = self.meta[sim]
            similars[sim]['distance'] = dist
        return(similars)
    
    def get_movie_meta(self):
        print("Refreshing metadata")
        nebula_movies={}
        query = 'FOR doc IN Movies RETURN doc'
        cursor = self.db.aql.execute(query)
        for data in cursor: 
            #print(data)
            nebula_movies[data['_id']] = data
        return(nebula_movies) 
    
    def get_scenes(self):
        nebula_movies={}
        query = 'FOR doc IN SemanticStoryNodes FILTER  RETURN doc'
        cursor = self.db.aql.execute(query)
        for data in cursor: 
            #print(data)
            nebula_movies[data['_id']] = data
        return(nebula_movies) 

    def get_stories_from_db(self):
        query = 'FOR doc IN Stories RETURN doc'
        stories = {}
        cursor = self.db.aql.execute(query)
        for data in cursor:
            #print(data)
            stories[data['movie_id']]=data
        return(stories)
    
    def get_bert_id_from_db(self, movie_id):
        query = 'FOR doc IN milvus_bert_embeddings_development FILTER doc.nebula_movie_id == "{}"  RETURN doc'.format(movie_id)
        cursor = self.db.aql.execute(query)
        vectors = []
        for data in cursor:
            #print("Query sentence: ", data['sentence'])
            vectors.append(int(data['milvus_key']))
        return(vectors)
    
    def get_doc2vec_id_from_db(self, movie_id):
        query = 'FOR doc IN milvus_doc2vec_embeddings FILTER doc.nebula_movie_id == "{}"  RETURN doc'.format(
            movie_id)
        cursor = self.db.aql.execute(query)
        vectors = []
        for data in cursor:
            #print("Query sentence: ", data['sentence'])
            vectors.append(int(data['milvus_key']))
        return(vectors)

    def get_scene_id_from_db(self, movie_id, scene):
        fps = 28 
        scene_int = int(float(scene))
        scene_sec = scene_int * fps
        query = 'FOR doc IN milvus_scenes_hollywood_mean FILTER doc.nebula_movie_id == "{}"  RETURN doc'.format(movie_id)
        cursor = self.db.aql.execute(query)
        vectors = []
      
        for data in cursor:
            interval = data['sentence']
            _start, _stop = interval.split("-")
            start = int(_start)
            stop = int(_stop)
            #print("DEBUG: ",data)
            if scene_sec >= start and scene_sec <= stop:
                #print ("Start/stop: ", start, " ", stop )
                vectors.append(int(data['milvus_key']))
        return(vectors)

    def load_doc2vec_embeddings(self):
        embedding_dimensions = 80
        num_index_trees = 512
        single_index = SimpleNeighbors(embedding_dimensions)    
        for key in self.stories.values():
            #print(key)
            embedding = self.model.docvecs[key['movie_id']]
            #print(embedding)
            single_index.add_one(key['movie_id'],embedding)
        single_index.build(n=num_index_trees) 
        return(single_index)

    def get_all_vectors(self):
        embeddings = []  
        for key in self.stories.values():
            #print(key)
            embedding = self.model.docvecs[key['movie_id']]
            embeddings.append(embedding)
        return(embeddings)

    def clip_text_encoder(self, _text):
        text_inputs = torch.cat([clip.tokenize(_text)]).to(self.device)
        with torch.no_grad():
            text_features = self.clip.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            vector = text_features.tolist()[0]
        return(vector)
    
    def get_all_scenes_of_movie(self, movie_id):
        query = 'FOR doc IN milvus_scenes_hollywood_mean FILTER doc.nebula_movie_id == "{}"  RETURN doc'.format(movie_id)
        #print(scene_sec)
        cursor = self.db.aql.execute(query)
        vectors_idx = []
      
        for data in cursor:
            interval = data['sentence']
            _start, _stop = interval.split("-")
            start = int(_start)
            stop = int(_stop)
            vectors_idx.append(int(data['milvus_key']))
        return(vectors_idx)

    def get_similar_scenes(self, movie_id, scene, top):
        similars = {}
        self.meta = self.get_movie_meta()
        print("Find similar scene for: ", movie_id, "search method: ", self.type)
        print("Clip based search...")
        vector_idx = self.get_scene_id_from_db(movie_id, scene)
        #print(vector_idx)
        if len(vector_idx) == 0:
            return (None)
            # sims = self.index.get_vector_by_id(vectors)
        status, sim = self.index.get_vector_by_id(vector_idx[0])
        search_ = self.index.search_vector(top, sim)
        for _data in search_:
            data = _data[1]
            dist = _data[0]
            #print(data['nebula_movie_id'], " ", data['sentence'], " ", dist)
            similars[data['nebula_movie_id']] = self.meta[data['nebula_movie_id']]
            similars[data['nebula_movie_id']]['scene_meta'] = data
            similars[data['nebula_movie_id']]['distance'] = 1/dist
            #print("Meta: ",self.meta[data['nebula_movie_id']])
        #print("SIMS:", similars)
        return(similars)
    
    def get_similar_movies(self, movie_id, top):
        similars = {}
        self.meta = self.get_movie_meta()
        print("Find similar for: ", movie_id, "search method: ", self.type)
        if self.type == 'doc2vec':
            print("Doc2Vec based search...")
            vectors = self.get_doc2vec_id_from_db(movie_id)
            if len(vectors) == 0:
                return (None)
            #print(vectors)
            # sims = self.index.get_vector_by_id(vectors)
            status, sim = self.index.get_vector_by_id(vectors[0])
            search_ = self.index.search_vector(top, sim)
            for _data in search_:
                data = _data[1]
                dist = _data[0]
                similars[data['nebula_movie_id']
                         ] = self.meta[data['nebula_movie_id']]
                similars[data['nebula_movie_id']]['distance'] = dist
                #print("Meta: ",self.meta[data['nebula_movie_id']])
            #print("NEB EMB API: ", similars)
            return(similars)
        elif self.type == 'clip2bert':
            print("Bert based search...")
            vectors = self.get_bert_id_from_db(movie_id)
            if len(vectors) == 0:
                return (None)
            # print(vectors)
            # sims = self.index.get_vector_by_id(vectors)
            status, sim = self.index.get_vector_by_id(vectors[0])
            search_ = self.index.search_vector(top, sim)
            for _data in search_:
                data = _data[1]
                dist = _data[0]
                #print(data['nebula_movie_id'], " ", data['sentence'], " ", dist)
                similars[data['nebula_movie_id']] = self.meta[data['nebula_movie_id']]
                similars[data['nebula_movie_id']]['distance'] = dist
                #print("Meta: ",self.meta[data['nebula_movie_id']])
            # print("SIMS:", similars)
            return(similars)
        
        elif self.type == 'clip2scene':
           print("Please call get scenes foo")
            
        elif self.type == 'clip4string':
            print("CLIP4STRING based similarity search")
            # find the appropriate movie id
            if movie_id in self.sim_meta_data[0]:
                row_id = self.sim_meta_data[0].index(movie_id)
            else:
                return (None)
            sim_row = copy.deepcopy(self.sim_matrix[row_id, :])
            done = False
            cnt = 0

            for k in range(top):
                best_match = np.argmax(sim_row)
                dist = sim_row[best_match]
                similars[self.sim_meta_data[0][best_match]] = self.meta[self.sim_meta_data[0][best_match]]
                similars[self.sim_meta_data[0][best_match]]['distance'] = dist
                sim_row[best_match] = 0
            # print(similars)
            return(similars)


   
    #сdef set_embedding_type(embeddings_type):

    def __init__(self, embeddings_type, debug=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, self.preprocess = clip.load("RN50x4", self.device)
        if not debug:
            self.cfg = Cfg(['kfchannel','graphdb'])
        connect = DatabaseConnector()
        self.type = embeddings_type
        self.nre = NRE_API()
        self.db = self.nre.db
        self.database = self.nre.database
        if embeddings_type == 'doc2vec':
            print("Doc2Vec based similarity search")
            self.meta = self.get_movie_meta()
            #print("Metadata loaded")
            # self.bert = SentenceTransformer('paraphrase-mpnet-base-v2')
            self.encoder = GraphEncoder()
            self.index = MilvusAPI(
                'milvus', 'doc2vec_embeddings', self.database, 640)
            print("Milvus API loaded")
    
        elif embeddings_type == 'clip2bert':
            print("Bert based similarity search") 
            self.meta = self.get_movie_meta()
            #print("Metadata loaded")
            # self.bert = SentenceTransformer('paraphrase-mpnet-base-v2')
            self.encoder = GraphEncoder()
            print("Encoder loaded")
            self.index = MilvusAPI('milvus', 'bert_embeddings_development', self.database, 768)
            print("Milvus API loaded")

        elif embeddings_type == 'clip2scene':
            print("Clip based scene search")
            #self.clip_encoder = NebulaVideoEvaluation()
            self.meta = self.get_movie_meta()
            self.index = MilvusAPI('milvus', 'scenes_hollywood_mean', self.database, 640)

        elif embeddings_type == 'clip4string':
            print("Clic based string-like similarity search")
            #self.db = connect.init_nebula_db(self.cfg.get('graphdb','name'))
            self.meta = self.get_movie_meta()
            self.sim_meta_data = pd.read_pickle("/movies/data/clip4string_metadata.pickle")
            self.sim_matrix = np.load("/movies/data/sim_benchmark.npy")


