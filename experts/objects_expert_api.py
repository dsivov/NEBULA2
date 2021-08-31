from arango import ArangoClient
import logging
from nebula_api.nebula_enrichment_api import NRE_API

class OBJECTS_EXPERT_API:
    def __init__(self):
        self.nre = NRE_API()
        self.db = self.nre.db
        
    #When insert actor_id = description
    #Update will change description only (ReID)
    #Example 
    # insert person3 -> actor_id = person3, description person3
    # ReID update -> actor_id = person3, description person1 (If ReID found person1 == person3)
    def create_actor(self, actor_id, description, entity_id, confidence, start_time_offset, end_time_offset, bboxs, movie_id, arango_id, fps):
        start_frame = int(float(start_time_offset) * fps)
        end_frame = int(float(end_time_offset) * fps)
        query = 'UPSERT { movie_id: @movie_id, arango_id: @arango_id, \
            actor_id: @actor_id, entity_id: @entity_id}\
                 INSERT  \
            { movie_id: @movie_id, arango_id: @arango_id, actor_id: @actor_id, description: @description, entity_id: @entity_id, class: "Actors", confidence: @confidence,\
            start_time_offset: @start_time_offset, \
            end_time_offset: @end_time_offset, bboxs: @bboxs, updates: 1}\
                 UPDATE \
                { updates: OLD.updates + 1, description: @description} IN Nodes \
                    RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
        bind_vars = {
                        'movie_id': movie_id,
                        'arango_id': arango_id,
                        'entity_id': entity_id,
                        'actor_id': actor_id,
                        'description': description,
                        'confidence': confidence, 
                        'start_time_offset': start_time_offset, 
                        'end_time_offset': end_time_offset,
                        'start_frame': start_frame,
                        'end_frame': end_frame,      
                        'bboxs': bboxs #Tracking info, array of bounding boxes
                        }
        self.db.aql.execute(query, bind_vars=bind_vars)
