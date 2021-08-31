
from sys import prefix
from gensim.models.doc2vec import TaggedDocument

from multiprocessing.managers import all_methods
import networkx as nx

from numpy.lib.function_base import append, average
from scipy.ndimage.measurements import label
from scipy.stats.stats import mode
from embeddings.nebula_networkx_adapter import Nebula_Networkx_Adapter
from arango import ArangoClient

import sys

def connect_db(dbname):
    #client = ArangoClient(hosts='http://ec2-18-219-43-150.us-east-2.compute.amazonaws.com:8529')
    client = ArangoClient(hosts='http://18.159.140.240:8529')
    db = client.db(dbname, username='nebula', password='nebula')
    return (db)

# Specify attributes to be imported
def nebula_get_graph_formdb(ma, _filter):
    attributes = { 'vertexCollections':
                                    {'Actors': {'labels','description'},'Actions': {'labels','description'},
                                    'Relations': {'labels','description'}, 'Properties':{'labels','description'}},
                    'edgeCollections' :
                                    {'ActorToAction': {'_from', '_to','labels'},'ActorToRelation': {'_from', '_to','labels'}, 
                                    'MovieToActors':{'_from', '_to', 'labels'}, 'RelationToProperty':{'_from', '_to', 'labels'}}}

    # Export networkX graph  
    _filter = _filter                     
    g, lables, descriptions = ma.create_nebula_graph(graph_name = 'Test',  graph_attributes = attributes, _filter = _filter)
    fitG = nx.convert_node_labels_to_integers(g, first_label=0)
    return fitG, lables, descriptions

def nebula_get_stories(all_movies,ma):
    story = 0
    documents = []
    tags = {}
    nebula_metadata = {}
    for movie in all_movies.values():
        fitG, lables, descriptions = nebula_get_graph_formdb(ma, movie['movie']['movie_id'])
        prefix_labels = [] 
        stories = []
        successors = list(nx.dfs_preorder_nodes(fitG)) 
    
        for successor in successors:
            attrebutes = ""
            #print(fitG.nodes[successor]['attr_dict']['_class'], successor)
            nebs = dict(nx.bfs_successors(fitG, successor))
            #print(nebs)
            if lables[successor]:
                if int(lables[successor][0]) > 0 and ((int(lables[successor][2]) - int(lables[successor][1])) > 3):
                    if  fitG.nodes[successor]['attr_dict']['_class'] == "person" or fitG.nodes[successor]['attr_dict']['_class'] == "car":
                                                stories.append(fitG.nodes[successor]['attr_dict']['_object'])
                                                stories.append(fitG.nodes[successor]['attr_dict']['_object'] + "_from_" + lables[successor][1] + "_to_" + lables[successor][2]) 
                                            
            if (len(nebs) > 2):
                for neb in nebs[successor]:
                    if neb in nebs:
                        for next_neb in nebs[neb]:
                            if (fitG.nodes[successor]['attr_dict']['_class'] != fitG.nodes[next_neb]['attr_dict']['_class']):
                                _prefix = fitG.nodes[successor]['attr_dict']['_class'].replace(" ", "_").replace("(",'').replace(")", '').replace(".",'').replace(",",'')
                                _base = fitG.nodes[neb]['attr_dict']['_class'].replace(" ", "_").replace("(",'').replace(")", '').replace(".",'').replace(",",'')
                                _suffix = fitG.nodes[next_neb]['attr_dict']['_class'].replace(" ", "_").replace("(",'').replace(")", '').replace(".",'').replace(",",'')
                                sentence = _prefix + "_" + _base + "_" + _suffix
                                if ("Then" not in sentence) or ("With" not in sentence):
                                    #stories.append(sentence)
                                    stories.append(_prefix)
                                    stories.append(_base)
                                    stories.append(_suffix)
            _tag =  "story_" + str(story)
            dfs_doc = TaggedDocument(words= stories, tags=[_tag])
        
        # print(movie['movie']['_id'])
        # print("DFS-based neibs")
        # print(_tag)
        # print (len(stories))
        # print(dfs_doc)
        # input("Press Enter to continue...")
        documents.append(dfs_doc)
        tags[story]= _tag
        nebula_metadata[story] = (movie['movie']['file_name'], movie['movie']['_id'])
        print(nebula_metadata[story], story)
        story = story + 1
        
    print("Number of stories:", story)
    #print(documents)
    return(documents, tags, nebula_metadata)

def save_stories(db, nebula_meta, story):
    stories_col = db.collection('Stories')
    for i in nebula_meta:
        stories_col.insert(
            {
                'movie_id':nebula_meta[i][1],
                'movie_name': nebula_meta[i][0],
                'story': story[i]
            })


def main():
    # Specify the connection to the ArangoDB Database
    if len(sys.argv) < 2:
        print("Usage: ", sys.argv[0], " db_name")
        exit()
    db_name = sys.argv[1]
    con = {'dbName': db_name,
    'username': 'nebula',
    'password': 'nebula',
    'hostname': '18.159.140.240',
    'protocol': 'http', 
    'port': 8529}

    # Create Adapter instance
    ma = Nebula_Networkx_Adapter(conn = con) 
    db = connect_db(db_name)
    if db.has_collection('Stories'):
        db.delete_collection('Stories')
    db.create_collection('Stories')
    all_movies = ma.nebula_get_all_movies()
    stories, _tags, nebula_meta = nebula_get_stories(all_movies, ma)
    save_stories(db, nebula_meta, stories)  

if __name__ == '__main__':
    main()