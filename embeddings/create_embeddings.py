
from arango import ArangoClient
from simpleneighbors import SimpleNeighbors
from embeddings.nebula_model import NEBULA_DOC_MODEL
from gensim.models.doc2vec import TaggedDocument
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import random

import numpy as np

def connect_db(dbname):
    #client = ArangoClient(hosts='http://ec2-18-219-43-150.us-east-2.compute.amazonaws.com:8529')
    client = ArangoClient(hosts='http://18.159.140.240:8529')
    db = client.db(dbname, username='nebula', password='nebula')
    return (db)

def get_stories_from_db(db):
    query = 'FOR doc IN Stories RETURN doc'  
    stories = {}
    cursor = db.aql.execute(
            query
        )
    for data in cursor:
        #print(data)
        stories[data['movie_id']]=data
        print ()
    #print (stories)
    return(stories)

def create_doc2vec_embeddings(stories):
    sentences = []
    tags = {}
    embedding_dimensions = 80
    
    single_index = SimpleNeighbors(embedding_dimensions)
    single_index_dm = SimpleNeighbors(embedding_dimensions)
    stories1 = list(stories.items())
    random.shuffle(stories1)
    stories = dict(stories1)

    for i, story in enumerate(stories.values()):  
        dfs_doc = TaggedDocument(words=story['story'][0], tags=[story['movie_id']])
        sentences.append(dfs_doc) 
        tags[i] = story['movie_id']
 
    _algo = 0
    _window = 5
    _epoch = 5000
    model = NEBULA_DOC_MODEL(algo = _algo, window = _window, dimensions=embedding_dimensions, epochs=_epoch)
    model.fit(sentences, tags)
    print(len(sentences), " ", len(tags))
    sentence_embeddings = model._get_embeddings(tags)
   
    db = DBSCAN(eps=2,  min_samples=2).fit(sentence_embeddings)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("DBOW")
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    kmeans = KMeans(n_clusters=int(len(sentences)/10), random_state=0).fit(sentence_embeddings)
    centers = kmeans.cluster_centers_
    
    for key in tags.values():
        embedding = model._get_single_embedding(key)
        single_index.add_one(key,embedding)
    return(single_index, centers)

def main():
    num_index_trees = 512
    db = connect_db("nebula_dev")
    stories = get_stories_from_db(db)
    index, centers = create_doc2vec_embeddings(stories)
    #analize_doc2vec_embeddings(stories)
    #index = create_bert_embeddings(stories)

    index.build(n=num_index_trees) 
    index.save("model/nebula_index_single")

if __name__ == '__main__':
    main()