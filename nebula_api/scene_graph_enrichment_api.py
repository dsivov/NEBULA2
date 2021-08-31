from arango import ArangoClient
import time
from nebula_api.nebula_enrichment_api import NRE_API 


class SCENE_GRAPH_EXPERT_API:
    def __init__(self):
        self.nre = NRE_API()
        self.db = self.nre.db
   
    def get_movie_meta(self, movie_id):
        nebula_movies={}
       
        query = 'FOR doc IN Movies FILTER doc._id == "{}"  RETURN doc'.format(movie_id)
        
        cursor = self.db.aql.execute(query)
        for data in cursor: 
            nebula_movies[data['_id']] = data
        return(nebula_movies) 

    def insert_node_to_scenegraph(self, movie_id, arango_id, _class, scene_element, description, start, stop):
        query = 'UPSERT { movie_id: @movie_id, description: @description, scene_element: @scene_element} INSERT  \
            { movie_id: @movie_id, arango_id: @arango_id, class: @class, description: @description, \
                 scene_element: @scene_element, start: @start, stop: @stop, step: 1} UPDATE \
                { step: OLD.step + 1 } IN Nodes \
                    RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
        bind_vars = {'movie_id': movie_id,
                        'arango_id': arango_id,
                        'class': _class,
                        'scene_element': scene_element, 
                        'start': start, 
                        'stop': stop,
                        'description': description
                        }
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        for doc in cursor:
            doc=doc
        return(doc['doc']['_id'])
    
    

    def insert_edge_to_scenegraph(self, movie_id, arango_id, description, _from, _to):
        query = 'UPSERT { movie_id: @movie_id, _from: @from, _to: @to} INSERT  \
            { movie_id: @movie_id, arango_id: @arango_id, description: @description, _from: @from, _to: @to, step: 1} UPDATE \
                { step: OLD.step + 1} IN Edges \
                    RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
        bind_vars = {'movie_id': movie_id,
                        'arango_id': arango_id,
                        'from': _from, 
                        'to': _to,
                        'description': description
                        }
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        for doc in cursor:
            doc=doc
        return(doc['doc']['_id'])
    
    def get_scenes(self, movie_id):
        query = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}"  RETURN doc'.format(movie_id)
        cursor = self.db.aql.execute(query)
        #print("Movie: ", m)
        all_scenes = []
        scene = {}
        for i, data in enumerate(cursor):
            #print("SCENE ELEM.: ", data['scene_'])
            scene = {'scene_graph_triplets': data['scene_graph_triplets'], \
                    'movie_id': data['movie_id'], 'arango_id': data['arango_id'], \
                    'description': data['description'], 'scene_element': data['scene_element'], 'start': data['start'], 'stop': data['stop'], \
                     }
            #print(scene)
            all_scenes.append(scene)
        return(all_scenes)

    def create_scene_graph(self, movie_id):
        #movie_meta = self.get_movie_meta(movie_id)
        scenes = self.get_scenes(movie_id)
        for scene in scenes:
            #print(scene)
            #print(scene['arango_id'])
            for scene_graph_triplets in scene['scene_graph_triplets']:
                for scene_graph in scene_graph_triplets:
                    #print(scene_graph)
                    _object = scene_graph[0]
                    _subject = scene_graph[2]
                    _relation = scene_graph[1]
                    object_id = self.insert_node_to_scenegraph(scene['movie_id'], scene['arango_id'], "Object", scene['scene_element'], \
                        _object, scene['start'], scene['stop'])
                    #time.sleep(1)
                    subject_id = self.insert_node_to_scenegraph(scene['movie_id'], scene['arango_id'], "Subject", scene['scene_element'], \
                        _subject, scene['start'], scene['stop'])
                    self.insert_edge_to_scenegraph(scene['movie_id'], scene['arango_id'], _relation, object_id, subject_id)
                    

        #print
        #print(scene_graphs)

def main():
    # movie_id = 'Movies/92354707'
    scene_graph = SCENE_GRAPH_EXPERT_API()
    # print("Processing Movie: ", movie_id)
    nre = NRE_API()
    movies = nre.get_all_movies()
    for movie in movies:
        scene_graph.create_scene_graph(movie)
    #clip.test_clip_vectors()
    #clip.get_sentences()
if __name__ == "__main__":
    main()

