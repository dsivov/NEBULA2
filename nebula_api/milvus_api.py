from arango import collection
import milvus
from numpy import delete
from nebula_api.databaseconnect import DatabaseConnector as dbc
from milvus import Milvus, IndexType, MetricType, Status
from config_nebula.config import NEBULA_CONF


def connect_db(dbname):
    arangodb = dbc()
    db = arangodb.connect_db(dbname)
    return db


class MilvusAPI():
    def __init__(self, backend , collection_name, dbname, dim):
        self.collection_name = collection_name
        self.db = connect_db(dbname)
        self.dbname = dbname
        self.dim = dim
        if self.db.has_collection('milvus_' + collection_name):
            milvus_metadata = self.db.collection('milvus_' + collection_name)
        else:
            milvus_metadata = self.db.create_collection('milvus_' + collection_name)
        self.milvus_metadata = milvus_metadata
        if backend == 'milvus':
            self.create_milvus_database(dim)
            self.backend = 'milvus'

    def create_milvus_database(self, dimmension):
        #_HOST = '172.31.7.226'
        #_PORT = '19530'  # default value
        ncfg = NEBULA_CONF()
        _HOST, _PORT = ncfg.get_milvus_server()
        # Vector parameters
        _DIM = dimmension  # dimension of vector

        _INDEX_FILE_SIZE = 320  # max file size of stored index

        self.milvus = Milvus(_HOST, _PORT)
        metadata = {}
        collection_name = self.collection_name
        self.index_param = {
        'nlist': 2048
        }
        status, ok = self.milvus.has_collection(collection_name)
        if not ok:
            param = {
                'collection_name': collection_name,
                'dimension': _DIM,
                'index_file_size': _INDEX_FILE_SIZE,  # optional
                'metric_type': MetricType.IP  # optional
            }
            self.milvus.create_collection(param) 
            status = self.milvus.create_index(self.collection_name, IndexType.ANNOY, self.index_param)
            print("Milvus Collection Loaded: ", collection_name)
        else:
            #self.milvus.load_collection(self.collection_name)
            print("Milvus Collection Loaded: ", collection_name)
    
    def insert_vectors(self, embeddings, metadata):
        if len(embeddings) == len(metadata):
            status, ids = self.milvus.insert(self.collection_name, embeddings)
            print(status)
            self.milvus.flush([self.collection_name])
            self.milvus.load_collection(self.collection_name)
            for idx, meta in zip(ids, metadata):
                nebula_data = {
                    'filename': meta['filename'],
                    'movie_id': meta['movie_id'],
                    'nebula_movie_id': meta['nebula_movie_id'],
                    'stage': meta['stage'],
                    'frame_number': meta['frame_number'],
                    'sentence': meta['sentence'],
                    'milvus_key': str(idx)
                    }
                self.milvus_metadata.insert(nebula_data)
        else:
            print("Metadata must have same lenght with embeddings: Meta - ", len(metadata), ", Embeddings - ", len(embeddings) )

    def search_vector(self, limit,  vector):
        #print(vector)
        #Debug 
        import time
        if self.backend == 'milvus':
            search_param = {
                "nprobe": 256
            }
            #start_time = time.time()
            self.milvus.load_collection(self.collection_name)
            status, row_results = self.milvus.search(self.collection_name, limit, [vector], params=search_param)
            # print(self.collection_name)
            #print("ROW: ", row_results)  
            search_result = []
            #print("Milvus time --- %s seconds ---" % (time.time() - start_time))
            for results in row_results:
                for result in results:
                    #print("DEBUG: ", str(result.id))
                    milvus_key = str(result.id)
                    query = 'FOR doc IN milvus_' + self.collection_name + ' FILTER doc.milvus_key == @milvus_key RETURN doc'
                    bind_vars = {'milvus_key': milvus_key}
                    #start_time = time.time()
                    cursor = self.db.aql.execute(query, bind_vars=bind_vars)
                    #print("Arango time --- %s seconds ---" % (time.time() - start_time))
                    for data in cursor:
                        #print("SEARCH VECTOR DEBUG: ", data)
                        search_result.append([result.distance,data])
                        
            return(search_result)

      
    def get_vector_by_id(self, vector_id):
        if self.backend == 'milvus':
            status, vectors = self.milvus.get_entity_by_id(self.collection_name, [0,int(vector_id)])
            #print("Debug: ", vectors)
        return(vectors)

    def drop_database(self):
        status, ok = self.milvus.has_collection(self.collection_name)
        if ok:
            self.milvus.drop_collection(self.collection_name) 
        if self.db.has_collection('milvus_' + self.collection_name):
            self.db.delete_collection('milvus_' + self.collection_name)
    
    def delete_movie_vector(self, movie_id, collection_name):
        query = 'FOR doc IN milvus_' + collection_name + \
            ' FILTER doc.movie_id == @movie_id RETURN doc'
        bind_vars = {'movie_id': movie_id}
        #start_time = time.time()
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        #print("Arango time --- %s seconds ---" % (time.time() - start_time))
        data = {}
        for data in cursor:
            meta = data
        # print(data)
        if data:
            print("Entity exists, delete before insert")
            milvus_id = [0, int(data['milvus_key'])]
            self.milvus.delete_entity_by_id(collection_name, milvus_id)
            delete_query = 'FOR doc IN milvus_' + collection_name + \
                ' FILTER doc.movie_id == @movie_id REMOVE doc IN  milvus_' + collection_name
            # print(delete_query)
            self.db.aql.execute(delete_query, bind_vars=bind_vars)
    
    def init_db(self):
        if self.db.has_collection('milvus_' + self.collection_name):
            milvus_metadata = self.db.collection('milvus_' + self.collection_name)
        else:
            milvus_metadata = self.db.create_collection('milvus_' + self.collection_name)
        self.milvus_metadata = milvus_metadata
        if self.backend == 'milvus':
            self.create_milvus_database(self.dim)
            self.backend = 'milvus'
