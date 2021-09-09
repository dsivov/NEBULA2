
from arango import ArangoClient
from simpleneighbors import SimpleNeighbors
from embeddings.nebula_model import NEBULA_DOC_MODEL
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import random
import gensim
from milvus_api.milvus_api import MilvusAPI


import numpy as np

def connect_db(dbname):
    #client = ArangoClient(hosts='http://ec2-18-219-43-150.us-east-2.compute.amazonaws.com:8529')
    #client = ArangoClient(hosts='http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:8529')
    client = ArangoClient(hosts='http://172.31.11.112:8529')

    db = client.db(dbname, username='nebula', password='nebula')
    return (db)

def get_stories_from_db(db):
    stories = {}
    query_m = 'FOR doc IN Movies RETURN doc'
    cursor_m = db.aql.execute(query_m)
    for movie in cursor_m:
        story = []
        #print(movie['_id'])
        query = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}" RETURN doc'.format(movie['_id']) 
        cursor = db.aql.execute(
                query
            )
        for data in cursor:
            for a in data['scene_graph_triplets']:
                for b in a:
                    for w in b:
                        tokens = gensim.utils.simple_preprocess(w)
                        story= story + tokens
        #print(story)
        stories[movie['_id']]=story
        #input()
            
    #print (stories)
    return(stories)

def create_doc2vec_embeddings(stories):
    index = MilvusAPI('milvus', 'doc2vec_embeddings', 'nebula_datadriven', 640)
    sentences = []
    tags = {}
    embedding_dimensions = 640
    for i, story in enumerate(stories):  
        dfs_doc = TaggedDocument(words=stories[story], tags=[story])
        #print(dfs_doc)
        #input()
        sentences.append(dfs_doc) 
        tags[i] = story
    _algo = 0
    _window = 5
    _epoch = 1000
    model = NEBULA_DOC_MODEL(algo = _algo, window = _window, dimensions=embedding_dimensions, epochs=_epoch)
    model.fit(sentences, tags)
    #print(len(sentences), " ", len(tags))
    #sentence_embeddings = model._get_embeddings(tags)
   
    for key in tags.values():
        #print(key)
        #input()
        embedding = model._get_single_embedding(key)
        vector = embedding.tolist()
        #print(vector)
        #input()
        meta = {
                'filename': 'none',
                'movie_id': key,
                'nebula_movie_id': key,
                'stage': 'none',
                'frame_number': 'none',
                'sentence': 'none',
                }
        index.insert_vectors([vector], [meta])
        


def main():
    db = connect_db("nebula_datadriven")
    stories = get_stories_from_db(db)
    create_doc2vec_embeddings(stories)
    #analize_doc2vec_embeddings(stories)
    #index = create_bert_embeddings(stories)

    #index.build(n=num_index_trees) 
    #index.save("model/nebula_index_single")

if __name__ == '__main__':
    main()