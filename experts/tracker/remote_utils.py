import os
import sys

from arango import ArangoClient
import boto3

# import from nebula API
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from nebula_api.nebula_enrichment_api import NRE_API


class RemoteUtility:
    """
    A utility class for communicating with DB and web services.
    """

    def __init__(self):
        """
        setup remote utility clients and tools.
        """
        self.download_bucket_name = "nebula-frames"
        self.s3 = boto3.client('s3', region_name='eu-central-1')
        self.nre = NRE_API()
        self.client = ArangoClient(hosts='http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:8529')
        self.db = self.client.db("nebula_development", username='nebula', password='nebula')

    def downloadDirectoryFroms3(self, arango_id):
        """
        Download a movie's directory of frames.
        @param: arango_id: movie ID including path, e.g. Movies/12345678
        @return: the number of downloaded frames
        """
        # prepare download bucket
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(self.download_bucket_name)

        # iterate download items
        keys = []
        for i, obj in enumerate(bucket.objects.filter(Prefix=arango_id)):
            if i == 0:
                continue  # first output is the directory
            keys.append(obj.key)

            if os.path.isfile(obj.key):  # frame already exists. skip download.
                continue

            if not os.path.exists(os.path.dirname(obj.key)):
                os.makedirs(os.path.dirname(obj.key))
            bucket.download_file(obj.key, obj.key)  # save to same path
        
        return len(keys)

    def save_track_data_to_scenegraph(self, arango_id, track_data):
        """
        save all tracking data into the scene graphs DB, every object individually.
        @param: arango_id: movie ID including path, e.g. Movies/12345678.
        @param: track_data: an `autotracker` tracking output dictionary.
        @return: number of nodes added to the DB.
        """
        # get scenes info
        scenes = self.get_scenes(arango_id)

        # iterate all objects
        nodes_saved = 0
        for oid, data in track_data.items():

            # iterate intersecting scenes.
            for scene, start, stop in self.scene_intersection(scenes, data['start_frame'],
                                                              data['stop_frame']):
                self.insert_node_to_scenegraph(
                    scene['movie_id'],           # Found in scenes
                    scene['arango_id'],          # E.g. Movies/12345678
                    data['boxes'],               # dict format frame_id --> box(x, y, x, y)
                    data['scores'],              # dict format frame_id --> score
                    scene['scene_element'],      # intersecting scene element
                    data['class'] + str(oid),    # person0, car1, mango2, ..., sunglasses47, ...
                    start,                       # start of track (object found)
                    stop                         # end of track (object lost)
                )
                
                nodes_saved += 1
        
        return nodes_saved

    def scheduler_loop(self):
        """
        loops forever, waiting for movie tasks from arang.
        @return: yields movie arago ID's.
        """
        while True:
            # Signaling your code, that we have newly uploaded movie, frames are stored in S3.
            # Returns movie_id in form: Movie/<xxxxx>
            movies = self.nre.wait_for_change("Actors", "ClipScene") 
            for movie in movies:
                yield movie
            self.nre.update_expert_status("Actors") #Update scheduler, set it to done status

    def get_scenes(self, arango_id):
        """
        Get scenes for movie , you can find related scene element by comparing your start/stop and start/stop from database.
        @param: arango_id: movie ID including path, e.g. Movies/12345678
        """
        # query DB for all relevant scenes
        query = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}"  RETURN doc'.format(arango_id)
        cursor = self.db.aql.execute(query)
        
        # iterate DB output and save scenes info.
        all_scenes = []
        for data in cursor:
            all_scenes.append({
                'scene_graph_triplets': data['scene_graph_triplets'],  # e.g. "horse is brown"
                'movie_id': data['movie_id'],                          # random identifier
                'arango_id': data['arango_id'],                        # same as param
                'description': data['description'],                    # scene description
                'scene_element': data['scene_element'],                # index of scene scene part
                'start': data['start'],                                # scene element start frame
                'stop': data['stop']                                   # scene element stop frame
            })

        return all_scenes

    def insert_node_to_scenegraph(self, movie_id, arango_id, _bboxes, _scores, scene_element, description,
                                  start, stop):
        """
        Insert your data into database
        insert for every object!!
        @param: movie_id: a unique identifier hash for the movie.
        @param: arango_id: the path in the arango DB to the movie (E.g. Movies/12345678)
        @param: _bboxes: the bounding boxes per frame of the added object. This is a dictionary of lists of
                         the form: frame_id --> [xmin, ymin, xmax, ymax]).
        @param: _scores: the confidence scores per frame of the added object. This is a dictionary of lists
                         of the form: frame_id --> score
        @param: scene_element: the scene element the object appears in.
        @param: description: a description of the object, of the form "<class_name><obj_id>", e.g. "car13"
        @param: start: The fame number of the first frame in which the object appears in this element.
        @param: stop: The frame at which the object disappears in this scene element.
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
            'bboxes': _bboxes,  # format xyxy
            'scores': _scores,  # array of floats
            'scene_element': scene_element,
            'start': start,  # can be on multiple scenes
            'stop': stop,  
            'description': description # Put here your class name + id,  "face1", "man2", "person16".... 
        }

        self.db.aql.execute(query, bind_vars=bind_vars)

    def scene_intersection(self, scenes, obj_start, obj_stop):
        """
        find all scene elements that intersect with the object's presence.
        @param: scenes: a list of scene dictionaries gotten from the `get_scenes` function.
        @param: obj_start: The fame number of the first frame in which the object appears in this element.
        @param: obj_stop: The frame at which the object disappears in this scene element.
        @return: a list of tuples (scene, start, stop) where the scenes are those that intersect with the
                 current object in time, and start and stop are the latest starting frame and earliest
                 stopping frame of either the scene element or the object.
        """
        # intersection cases:
        # 1) scene_start <= obj_start <= scene_stop <= obj_stop  -----X---Y----X--Y--
        # 2) obj_start <= scene_start <= obj_stop <= scene_stop  --Y--X---Y----X-----
        # 3) scene_start <= obj_start <= obj_stop <= scene_stop  -----X---Y--Y-X-----
        # 4) obj_start <= scene_start <= scene_stop <= obj_stop  --Y--X--------X--Y--
        return [
            (scene,
             max(obj_start, scene['start']),  # intersection start
             min(obj_stop, scene['stop']))    # intersection end
            for scene in scenes
            if (scene['start'] <= obj_start <= scene['stop'] or  #  covers cases (1) and (3)
                scene['start'] <= obj_stop <= scene['stop']  or  #  covers case (2)
                (obj_start <= scene['start'] <= obj_stop) and (obj_start <= scene['stop'] <= obj_stop))  # (4)
        ]
