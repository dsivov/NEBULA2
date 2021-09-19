import logging
from arango import ArangoClient
from nebula_api.nebula_enrichment_api import NRE_API



class STORY_GRAPH_EXPERT_API:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            level=logging.INFO)
        self.nre = NRE_API()
        self.db = self.nre.db

    def get_movie_meta(self, movie_id):
        nebula_movies={}

        query = 'FOR doc IN Movies FILTER doc._id == "{}"  RETURN doc'.format(movie_id)

        cursor = self.db.aql.execute(query)
        for data in cursor:
            nebula_movies[data['_id']] = data
        return(nebula_movies)

    def insert_node_to_storygraph(self, movie_id, arango_id, _class, scene_element, description, start, stop):
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



    def insert_edge_to_storygraph(self, movie_id, arango_id, description, _from, _to):
        query = 'UPSERT { movie_id: @movie_id, description: @description, _from: @from, _to: @to} INSERT  \
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

    def get_objects(self, movie_id, scene_element):
        query = 'FOR doc IN Nodes FILTER doc.class == \"Object\" AND doc.arango_id == "{}" AND doc.scene_element == {} RETURN doc'.format(
            movie_id, int(scene_element))
        #print(query)
        _objects = []
        cursor = self.db.aql.execute(query)
        for node in cursor:
            _objects.append(node)
        return(_objects)

    def get_scenes(self, movie_id):
        query = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}"  RETURN doc'.format(movie_id)
        cursor = self.db.aql.execute(query)
        #print("Movie: ", m)
        all_scenes = []
        scene = {}
        for i, data in enumerate(cursor):
            #print("SCENE ELEM.: ", data['scene_'])
            scene = {'_id': data['_id'], 'scene_graph_triplets': data['scene_graph_triplets'], \
                    'movie_id': data['movie_id'], 'arango_id': data['arango_id'], \
                    'description': data['description'], 'scene_element': data['scene_element'], 'start': data['start'], 'stop': data['stop'], \
                     }
            #print(scene)
            all_scenes.append(scene)
            all_scenes.sort(key=lambda x: x['scene_element'])
        #print(all_scenes)
        return(all_scenes)

    def create_story_graph(self, movie_id):
        #movie_meta = self.get_movie_meta(movie_id)
        scenes = self.get_scenes(movie_id)
    
        prev_scene = None
        for scene in scenes:
            print(scene)
            if prev_scene:
                _prev = prev_scene
                _next = scene['_id']
                self.insert_edge_to_storygraph(
                    scene['movie_id'], scene['arango_id'], "Then", _prev, _next)
            prev_scene = scene['_id']
            #print(_from,"   " ,_to)
            if scene['scene_element'] == 0:
                _from = movie_id
                _to = scene['_id']
                self.insert_edge_to_storygraph(
                    scene['movie_id'], scene['arango_id'], "", _from, _to)
            print(scene['arango_id'], scene['scene_element'])
            _objects = self.get_objects(movie_id, scene['scene_element'])
            for _object in _objects:
                _from = scene['_id']
                _to = _object['_id']
                self.insert_edge_to_storygraph(
                    scene['movie_id'], scene['arango_id'], "", _from, _to)

def main():
    movie_id = 'Movies/92354707'
    story_graph = STORY_GRAPH_EXPERT_API()
    nre = NRE_API()
    movies = nre.get_all_movies()
    for movie in movies:    
        print("Processing Movie: ", movie)
        story_graph.create_story_graph(movie)
    #clip.test_clip_vectors()
    #clip.get_sentences()
if __name__ == "__main__":
    main()

