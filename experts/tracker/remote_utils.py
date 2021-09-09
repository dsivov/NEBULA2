import os
import sys

from arango import ArangoClient
import boto3

# import from nebula API
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nebula_api.nebula_enrichment_api import NRE_API


class RemoteUtility:
    def __init__(self):
        self.download_bucket_name = "nebula-frames"
        self.s3 = boto3.client('s3', region_name='eu-central-1')
        self.nre = NRE_API()
        self.client = ArangoClient(hosts='http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:8529')
        self.db = self.client.db("nebula_development", username='nebula', password='nebula')

    def download_arango_by_id(self, arango_id):
        # boto3.set_stream_logger('botocore', level='DEBUG')
        self.downloadDirectoryFroms3(arango_id)

    def downloadDirectoryFroms3(self, remoteDirectoryName):
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(self.download_bucket_name)
        keys = []
        for i, obj in enumerate(bucket.objects.filter(Prefix=remoteDirectoryName)):
            if i == 0:
                continue  # first output is the directory
            keys.append(obj.key)

            if os.path.isfile(obj.key):  # frame already exists
                continue

            if not os.path.exists(os.path.dirname(obj.key)):
                os.makedirs(os.path.dirname(obj.key))
            bucket.download_file(obj.key, obj.key)  # save to same path
        
        return len(keys)

    def save_track_data_to_scenegraph(self, arango_id, track_data):
        nodes_saved = 0
        scenes = self.get_scenes(arango_id)
        for oid, data in track_data.items():
            for scene, start, stop in self.scene_intersection(scenes, data['start_frame'],
                                                              data['stop_frame']):
                self.insert_node_to_scenegraph(
                    scene['movie_id'],           # Found in scenes
                    scene['arango_id'],          # E.g. Movies/12345678
                    data['boxes'],               # dict format frame_id --> box(x, y, x, y)
                    data['scores'],              # dict format frame_id --> score
                    scene['scene_element'],      # TODO: fix this
                    data['class'] + str(oid),    # person0, car1, mango2, ..., sunglasses47, ...
                    start,                       # start of track (object found)
                    stop                         # end of track (object lost)
                )
                
                nodes_saved += 1
        
        return nodes_saved

    def scheduler_loop(self):
        while True:
            # Signaling your code, that we have newly uploaded movie, frames are stored in S3.
            # Returns movie_id in form: Movie/<xxxxx>
            movies = self.nre.wait_for_change("Actors", "ClipScene") 
            for movie in movies:
                yield movie
            self.nre.update_expert_status("Actors") #Update scheduler, set it to done status

    def get_scenes(self, arango_id):
        """
        Get scenes for movie , you can find related scene element by comparing your start/stop and start/stop from database
        """
        query = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}"  RETURN doc'.format(arango_id)
        cursor = self.db.aql.execute(query)
        #print("Movie: ", m)
        
        all_scenes = []
        scene = {}
        for i, data in enumerate(cursor):
            #print("SCENE ELEM.: ", data['scene_'])
            scene = {'scene_graph_triplets': data['scene_graph_triplets'],
                     'movie_id': data['movie_id'], 'arango_id': data['arango_id'],
                     'description': data['description'], 'scene_element': data['scene_element'],
                     'start': data['start'], 'stop': data['stop']}

            #print(scene)
            all_scenes.append(scene)

        return all_scenes

    def insert_node_to_scenegraph(self, movie_id, arango_id, _bboxes, _scores, scene_element, description,
                                  start, stop):
        """
        Insert your data into database
        insert for every object!!
        @param: movie_id: a unique identifier hash for the movie.
        @param: arango_id: the path in the arango DB to the movie (E.g. Movies/12345678)
        @param: _bboxes: 
        @param: _cores:
        @param: scene_element:
        @param: description: 
        @param: start:
        @param: stop: 
        """
        query = """UPSERT { 
                       movie_id: @movie_id, description: @description, scene_element: @scene_element
                   } INSERT {
                       movie_id: @movie_id, arango_id: @arango_id, class: 'Object',
                       description: @description, scores: @scores, bboxes: @bboxes, 
                       scene_element: @scene_element, start: @start, stop: @stop, step: 1
                   } UPDATE {
                       step: OLD.step + 1
                   } IN Nodes RETURN {
                       doc: NEW, type: OLD ? 'update' : 'insert'
                   }"""

        bind_vars = {
            'movie_id': movie_id,#From scenes
            'arango_id': arango_id, #in format of Movie/<xxxx>, what you will get from scheduler
            'bboxes': _bboxes,  # format xywh? xyxy? ask michael  array?
            'scores': _scores,  # array of floats
            'scene_element': scene_element,
            'start': start,  # can be on multiple scenes
            'stop': stop,  
            'description': description # Put here your class name + id,  "face1", "man2", "person16".... 
        }

        self.db.aql.execute(query, bind_vars=bind_vars)

    def scene_intersection(self, scenes, obj_start, obj_stop):
        return [
            (scene,
             max(obj_start, scene['start']),  # intersection start
             min(obj_stop, scene['stop']))    # intersection end
            for scene in scenes
            if (scene['start'] <= obj_start <= scene['stop'] or
                scene['start'] <= obj_stop <= scene['stop']  or
                (obj_start <= scene['start'] <= obj_stop) and ((obj_start <= scene['stop'] <= obj_stop)))
        ]
