import os
import sys

# import from common
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.RemoteAPIUtility import RemoteAPIUtility


class ActionsAPIUtility(RemoteAPIUtility):
    def create_action(self, arango_id, movie_id, frame_id, box, description, confidence, action_id):
        query = 'UPSERT { movie_id: @movie_id, arango_id: @arango_id, \
            action_id: @action_id}\
                 INSERT  \
            { movie_id: @movie_id, arango_id: @arango_id, actor_id: @actor_id, description: @description, frame_id: @frame_id, \
                class: "Actions", confidence: @confidence, updates: 1, box: @box}\
                 UPDATE \
                { updates: OLD.updates + 1, description: @description} IN Nodes \
                    RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
        bind_vars = {
            'movie_id': movie_id,
            'arango_id': arango_id,
            'frame_id': frame_id,
            'actor_id': None,
            'box': box,
            'description': description,     # string name of action
            'confidence': confidence,
            'action_id': action_id
        }
        self.db.aql.execute(query, bind_vars=bind_vars)

    def save_action_data_to_scene_graph(self, arango_id, actions_data):
        action_id = 0
        for frame_id, d in actions_data.items():
            for box, actions, scores in zip(d['detection_boxes'], d['detection_classes'], d['detection_scores']):
                for description, confidence in zip(actions, scores):
                    self.create_action(
                        arango_id=arango_id,
                        movie_id=self.get_movie_info(arango_id)['movie_id'],
                        frame_id=frame_id,
                        box=box,
                        description=description,
                        confidence=confidence,
                        action_id=action_id
                    )
                    action_id += 1

    def scheduler_loop(self):
        """
        loops forever, waiting for movie tasks from arang.
        @return: yields movie arago ID's.
        """
        while True:
            # Signaling your code, that we have newly uploaded movie, frames are stored in S3.
            # Returns movie_id in form: Movie/<xxxxx>
            movies = self.nre.wait_for_change("Actions", "ClipScene") 
            for movie in movies:
                yield movie
            self.nre.update_expert_status("Actions") #Update scheduler, set it to done status
