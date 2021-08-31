from arango import ArangoClient
import logging
from nebula_api.nebula_enrichment_api import NRE_API

class ACTIONS_EXPERT_API:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            level=logging.INFO)
        self.connect_db('nebula_datadriven')
        self.nre = NRE_API()
        self.db = self.nre.db

    def create_action(self, actor_id, action_id, description, entity_id, confidence, start_time_offset, end_time_offset, movie_id, arango_id, fps):
        start_frame = int(float(start_time_offset) * fps)
        end_frame = int(float(end_time_offset) * fps)
        query = 'UPSERT { movie_id: @movie_id, arango_id: @arango_id, \
            action_id: @action_id}\
                 INSERT  \
            { movie_id: @movie_id, arango_id: @arango_id, actor_id: @actor_id, action_id: @action_id, description: @description, entity_id: @entity_id, class: "Actions", confidence: @confidence,\
            start_time_offset: @start_time_offset, \
            end_time_offset: @end_time_offset, updates: 1}\
                 UPDATE \
                { updates: OLD.updates + 1, description: @description} IN Nodes \
                    RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
        bind_vars = {
                        'movie_id': movie_id,
                        'arango_id': arango_id,
                        'entity_id': entity_id,
                        'actor_id': actor_id,
                        'action_id': action_id,
                        'description': description,
                        'confidence': confidence, 
                        'start_time_offset': start_time_offset, 
                        'end_time_offset': end_time_offset,
                        'start_frame': start_frame,
                        'end_frame': end_frame
                        }
        self.db.aql.execute(query, bind_vars=bind_vars)
        #self.nre.update_expert_status("actions")
