from arango import ArangoClient
import json
from elasticsearch import Elasticsearch
import numpy as np


client = ArangoClient(hosts='http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:8529')
db = client.db("nebula_holywood2", username='nebula', password='nebula')

def extract_action(a):
      return ''.join(filter(lambda c: not c.isdigit(), a['belongs_to'])) +' '+ a['description']

def mid2actions(m):
      query_a = 'FOR doc IN Actions FILTER doc.movie_id == "{}" RETURN doc'.format(m)
      cursor_a = db.aql.execute(query_a)
      return '. '.join([extract_action(c) for c in cursor_a])

def extract_relation(r):
    return ' '.join([''.join(filter(lambda c: not c.isdigit(), s)) for s in r['annotator'].split()])

def mid2relations(m):
    query_r = 'FOR doc IN Relations FILTER doc.movie_id == "{}" RETURN doc'.format(m)
    cursor_r = db.aql.execute(query_r)
    return '. '.join([extract_relation(r) for r in cursor_r if 'annotator' in r])

def mid2sentences(m):
    query_r = 'FOR doc IN SemanticStoryNodes FILTER doc.movie_id == "{}"  AND doc.description == \'With\' RETURN doc.sentence'.format(m)
    cursor_r = db.aql.execute(query_r)
    #print("Movie: ", m)
    centenses = ""
    for i, sent_arry in enumerate(cursor_r):
        #print([sent_arry])
        for sents in [sent_arry]:
            for sent in sents:
                print("DEBUG: ", sent)
                centenses = centenses + sent
        # print("Movie: ", m ," Stage: ", i)
        # if len(sent_arry.split("\"")) > 6:
        #     if sent_arry.split("\"")[1] != sent_arry.split("\"")[3]:
        #         #print(sent_arry.split("\"")[1] + "." + sent_arry.split("\"")[3])
        #         centenses = centenses + "." + sent_arry.split("\"")[1] + "." + sent_arry.split("\"")[3]
        # elif len(sent_arry.split("\"")) > 1:
        #     #print(sent_arry.split("\"")[1])
        #     centenses = centenses + "." + sent_arry.split("\"")[1]
        # # for t in sent_arry.split("\""):
        # #     print (t)
    return centenses

def get_919_data():
    query = 'FOR doc IN Movies FILTER doc.split == \'0\' AND doc.splits_total == \'30\' RETURN doc'
    cursor = db.aql.execute(query)
    cc=[c for c in cursor]
    docs=[]
    for c in cc:
        m=c['movie_id']
        description = mid2actions(m)+'. '+mid2relations(m) + '. ' + mid2sentences(m)
        docs.append({'movie_name':c['file_name'], 'video':c['url_path'],
        'timestamp':'0', 'description':description, 
        'movie_time_begin' : '0', 'movie_time_end' : '001', 
        'confidence' : [], 'parents':'', 'db_id' : c['_id']})
        #print()
    #print(docs)
    return(docs)

def get_Holywood2_data():
    query = 'FOR doc IN Movies RETURN doc'
    cursor = db.aql.execute(query)
    cc=[c for c in cursor]
    docs=[]
    for c in cc:
        m=c['movie_id']
        description = mid2actions(m)+'. '+mid2relations(m) + '. ' + mid2sentences(m)
        docs.append({'movie_name':c['file_name'], 'video':c['url_path'],
        'timestamp':'0', 'description':description, 
        'movie_time_begin' : '0', 'movie_time_end' : '001', 
        'confidence' : [], 'parents':'', 'db_id' : c['_id']})
        #print()
    #print(docs)
    return(docs)
#mid2sentences('Movies/10715274')
docs = get_Holywood2_data()
for doc in docs:
    print(doc)
print(docs)
es_host='http://tnnb2_master:NeBuLa_2@http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:9200/'
es = Elasticsearch(hosts=[es_host])
response = es.index(
    index = 'holywood2',
    body = {'doc':doc}
)
assert response['result'] == 'created'