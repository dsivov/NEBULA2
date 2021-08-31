from nebula_api.milvus_api import MilvusAPI

from nebula_api.databaseconnect import DatabaseConnector
from nebula_api.graph_encoder import GraphEncoder

import random

import pickle
import numpy as np
from sklearn import preprocessing

gdb = DatabaseConnector()
#db = gdb.connect_db("nebula_development")

def encode_movie_to_bert(movie_aranngo_id, db):
    """
    Encode all scene elements of a movie
    :param movie_arango_id: for example, 'Movies/92349435'
    :param db_name: database name (nebula_datadriven)
    :return: torch.Size([1, 768]) + dictionary of meta data
    """
    query = f'FOR doc IN StoryLine FILTER doc.arango_id == \'{movie_aranngo_id}\' RETURN doc'
    cursor = db.aql.execute(query, ttl=3600)
    cur_sentences = []
    graph_encoder = GraphEncoder()
    for cnt, sent_array in enumerate(cursor):
        cur_sentences.append(sent_array)
        meta = {
            'filename': 'none',
            'movie_id': sent_array['arango_id'],
            'nebula_movie_id': sent_array['arango_id'],
            'stage': sent_array['scene_element'],
            'frame_number': 'none',
            'sentence': sent_array['sentences'],
        }
    final_sent = graph_encoder.process_scene_sentences(cur_sentences)

    if final_sent is not None:
        embedding = graph_encoder.encode_with_transformers(final_sent)
        meta['sentence'] = final_sent
    else:
        return None, None

    return embedding, meta


if __name__ == '__main__':
    print('Creating BERT embeddings')
    # debug_bert()
    # 92453176
    db = gdb.connect_db('nebula_development')
    embedding, meta = encode_movie_to_bert('Movies/96995651', db)
    print(embedding.tolist())
    print(meta)
