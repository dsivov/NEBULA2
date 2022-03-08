import random
import sys
import os
import time
import urllib
from arango import ArangoClient
from nebula_api.databaseconnect import DatabaseConnector
from config_nebula.config import NEBULA_CONF




class NRE_API:
    def __init__(self):
        #self.connect_db("nebula_development")
        config = NEBULA_CONF()
        self.db_host = config.get_database_host()
        self.database = config.get_database_name()
        gdb = DatabaseConnector()
        self.db = gdb.connect_db(self.database)
        self.es_host = config.get_elastic_host()
        self.index_name = config.get_elastic_index()
        self.gdb = gdb
        self.temp_file = "/tmp/video_file.mp4" 
    
    def get_new_movies(self):
        nebula_movies=[]
        #query = 'FOR doc IN Movies FILTER doc.split == \'0\' AND doc.splits_total == \'30\' RETURN doc'
        query = 'FOR doc IN Movies FILTER doc.status == \'created\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/10715274\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/12911567\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/12476045\' OR doc._id == \'Movies/12465271\' RETURN doc'
        cursor = self.db.aql.execute(query)
        for data in cursor: 
            nebula_movies.append(data['_id'])
        return(nebula_movies) 

    def get_all_movies(self):
        nebula_movies = []
        #query = 'FOR doc IN Movies FILTER doc.split == \'0\' AND doc.splits_total == \'30\' RETURN doc'
        query = 'FOR doc IN Movies RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/10715274\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/12911567\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/12476045\' OR doc._id == \'Movies/12465271\' RETURN doc'
        cursor = self.db.aql.execute(query)
        for data in cursor:
            nebula_movies.append(data['_id'])
        return(nebula_movies)
    
    def get_movie_url(self, movie_id):
        #nebula_movies = []
        url = ""
        #query = 'FOR doc IN Movies FILTER doc.split == \'0\' AND doc.splits_total == \'30\' RETURN doc'
        query = 'FOR doc IN Movies FILTER doc._id == "{}" RETURN doc.url_path'.format(movie_id)
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/10715274\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/12911567\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/12476045\' OR doc._id == \'Movies/12465271\' RETURN doc'
        cursor = self.db.aql.execute(query)
        for data in cursor:
            #print(data)
            url = data
        #print(url)
        return(url)
    
    def get_plugins(self):
        self.experts = []
        query = 'FOR doc IN nebula_experts RETURN doc'
        cursor = self.db.aql.execute(query)
        for data in cursor: 
            #print(data)
            self.experts.append(data)

    def register_plugin(self, port, _module, klass, filter):
        self.filter = filter
        self.port = port
        self._module = _module
        self.klass = klass

    def get_versions(self):
        versions = []
        query = 'FOR doc IN changes RETURN doc'
        cursor = self.db.aql.execute(query)
        for data in cursor: 
            #print(data)
            versions.append(data)
        return(versions)
    
    def get_expert_status(self, expert, depends):
        versions = self.get_versions()
        for version in versions:
            if version[depends] > version[expert]:
                #print(version)
                return True
            else:
                return False

    def wait_for_change(self, expert, depends):
        while True:
            if self.get_expert_status(expert, depends):
                movies = self.get_new_movies()
                #print("New movies: ", movies)
                return(movies)
            time.sleep(3)

    def wait_for_finish(self, experts):
        while True:
            versions = self.get_versions()
            count = len(experts)
            for version in versions:
                global_version = version['movies']
                print(version)
                for expert  in experts: 
                    if global_version != version[expert]:
                        break
                    else:
                        count = count - 1
            if count <= 0:
                return True
            time.sleep(3)

    def update_expert_status(self, expert):
        if expert == "movies": #global version
            txn_db = self.db.begin_transaction(read="changes", write="changes")
            print("Updating global version")
            query = 'FOR doc IN changes UPDATE doc WITH {movies: doc.movies + 1} in changes'
            txn_db.aql.execute(query)
            txn_db.transaction_status()
            txn_db.commit_transaction()
            return True
        else:
            txn_db = self.db.begin_transaction(read="changes", write="changes")
            query = 'FOR doc IN changes UPDATE doc WITH {' + expert + ': doc.movies} in changes'
            #print(query)
            txn_db.aql.execute(query)
            txn_db.transaction_status()
            txn_db.commit_transaction()
            return True    
    
    def force_start_expert(self, expert):
        print("Updating global version")
        query = 'FOR doc IN changes UPDATE doc WITH {' + expert + ': doc.'+ expert + ' - 1} in changes'
        #print(query)
        self.db.aql.execute(query)
    
    def change_status_movie(self, status, movie_id):
        query = 'FOR doc IN Movies FILTER doc._id == \'' + movie_id + '\' UPDATE doc WITH {status: \''+ status +'\' } in Movies'
        self.db.aql.execute(query)
    
    def get_all_expert_data(self, expert, movie_id):
        #Actions, Actors
        expert_data = []
        query = 'FOR doc IN Nodes FILTER doc.arango_id == \'' + movie_id + '\' AND doc.class == \'' + expert + '\'AND (HAS(doc,"bboxes") OR HAS(doc,"box")) RETURN doc'
        cursor = self.db.aql.execute(query)
        for node in cursor:
            expert_data.append(node) 
        return(expert_data)
    
    def get_clip_data(self, movie_id):
        clip_data = {}
        query = 'FOR doc IN StoryLine FILTER doc.arango_id == \'' + movie_id + '\' RETURN doc'
        cursor = self.db.aql.execute(query)
        for node in cursor:
            clip_data[node['scene_element']] = [node['sentences'], node['scene_graph_triplets']] 
        return(clip_data)

    def get_vcomet_data(self, movie_id):
        vcomet_data = []
        query = 'FOR doc IN nebula_vcomet_lighthouse_lsmdc_mean FILTER doc.movie == \'' + movie_id + '\' RETURN doc'
        cursor = self.db.aql.execute(query)
        for node in cursor:
            vcomet_data.append(node)
            # print(node)
        return(vcomet_data)
    
    def get_groundings_from_db(self, movie_id, scene_element):
        results = {}
        query = 'FOR doc IN nebula_comet2020_lsmdc_scored_v03 FILTER doc.movie_id == "{}" AND doc.scene_element == {} RETURN doc'.format(movie_id, scene_element)
        #print(query)
        cursor = self.db.aql.execute(query)
        for doc in cursor:
            results.update(doc)
        return (results)

    def get_scene_from_collection(self, movie_id, scene_element, collection):
        results = {}
        query = 'FOR doc IN {} FILTER doc.movie_id == "{}" AND doc.scene_element == {} RETURN doc'.format(collection,movie_id, scene_element)
        #print(query)
        cursor = self.db.aql.execute(query)
        for doc in cursor:
            results.update(doc)
        return (results)

    def get_stages(self, m):
        query_r = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}" RETURN doc'.format(m)
        cursor_r = self.db.aql.execute(query_r)
        stages = []
        for stage in cursor_r:
            stages.append(stage)
        return(stages)
        
    def download_video_file(self, movie):
        import cv2
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

#nre = NRE_API()
#nre.get_vcomet_data("Movies/114206264")
# while True:
#     topic = "NRE"
#     messagedata = "new_plugin"
#     print (topic, " ", messagedata)
#     time.sleep(1)
