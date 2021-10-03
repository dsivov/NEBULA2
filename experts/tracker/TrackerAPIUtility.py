import os
import sys

# import from common
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.RemoteAPIUtility import RemoteAPIUtility


class TrackerAPIUtility(RemoteAPIUtility):
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
            'movie_id': movie_id,       # From scenes
            'arango_id': arango_id,     # in format of Movie/<xxxx>, what you will get from scheduler
            'bboxes': _bboxes,          # format xyxy
            'scores': _scores,          # array of floats
            'scene_element': scene_element,
            'start': start,             # can be on multiple scenes
            'stop': stop,  
            'description': description  # Put here your class name + id,  "face1", "man2", "person16".... 
        }

        self.db.aql.execute(query, bind_vars=bind_vars)

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
