from milvus_api.milvus_api import MilvusAPI

from benchmark.connection import connect_db
from benchmark.graph_encoder import GraphEncoder
from benchmark.connection import nebula_connect
from benchmark.connection import get_all_movies
from sentence_transformers import SentenceTransformer
from gdb.databaseconnect import DatabaseConnector
from embeddings.nebula_networkx_adapter import Nebula_Networkx_Adapter
import random

from semantic_text_similarity.models import WebBertSimilarity

import pickle
import numpy as np
from sklearn import preprocessing
from transformers import AutoTokenizer, AutoModel
import torch
from common.cfg import Cfg


# #Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
#     sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#     return sum_embeddings / sum_mask
#
# class GraphEncoder:
#     """
#     The class can create embeddings from a graph representation of a scene
#     """
#     def __init__(self):
#         self.bert = SentenceTransformer('paraphrase-mpnet-base-v2')
#
#         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
#         self.model = AutoModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
#
#     def simple_graph_encoding(self, sent_list: list):
#         """
#         The simplest method - treat all sentences equally
#         :param: sent_list - list of sentences
#         :return:
#         """
#         embedding = self.bert.encode(sent_list)
#         return embedding
#
#     def encode_with_transformers(self, sent_list: list):
#         # Tokenize sentences
#         encoded_input = self.tokenizer(sent_list, padding=True, truncation=True, max_length=256, return_tensors='pt')
#
#         tokens = torch.empty(1, 0, dtype=torch.int64)
#         for sent in sent_list:
#             encoded_input = self.tokenizer(sent, padding=True, truncation=True, max_length=256, return_tensors='pt')
#             tokens = torch.cat((tokens, encoded_input['input_ids'][:, 1:-1]), dim=1)
#             tokens = torch.cat((tokens, torch.tensor([102]).reshape((1, 1))), dim=1)
#
#         encoded_input['input_ids'] = tokens
#         encoded_input['attention_mask'] = torch.tensor([1] * max(tokens.shape)).reshape((1, max(tokens.shape)))
#
#         # Compute token embeddings
#         with torch.no_grad():
#             model_output = self.model(**encoded_input)
#
#         # Perform pooling. In this case, mean pooling
#         sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
#
#         return sentence_embeddings

def process_single_movie(sentences, num_stages, proc_type):

    # sent_list = [None] * num_stages

    sent_list = []

    for sentence in sentences:
        if sentence['description'] == 'With':
            if type(sentence['sentence']) == list:
                continue
            splited = sentence['sentence'].split("\"")
            found = ''
            for split in splited:
                if split[0].isalpha() and len(split) > 4:
                    found = split
                    break
            if found != '':
                sent_list.append(found)

    return sent_list

    # for sentence in sentences:
    #     if sentence['description'] == 'With':
    #         if type(sentence['sentence']) == list:
    #             return sentence['sentence']
    #         splited = sentence['sentence'].split("\"")
    #         found = ''
    #         for split in splited:
    #             if split[0].isalpha() and len(split) > 4:
    #                 found = split
    #                 break
    #         sent_list[sentence['stage']] = found
    #
    # if proc_type == 'full':
    #     final_sentence = sent_list[0]
    #     for cnt, s in enumerate(sent_list):
    #         if cnt == 0:
    #             continue
    #         final_sentence = final_sentence + ' then ' + s
    # elif proc_type == 'one':
    #     final_sentence = sent_list[0]
    #
    # elif proc_type == 'separate':
    #     final_sentence = [sent_list[0]]
    #     for cnt, s in enumerate(sent_list):
    #         if cnt == 0:
    #             continue
    #         final_sentence.append(s)
    #
    # return final_sentence


def encode_movie_to_bert(movie_aranngo_id, db_name):
    """
    Encode all scene elements of a movie
    :param movie_arango_id: for example, 'Movies/92349435'
    :param db_name: database name (nebula_datadriven)
    :return: torch.Size([1, 768]) + dictionary of meta data
    """
    db = connect_db(db_name)
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

def get_sentences_from_movies(db_name, proc_type='full'):
    db = connect_db(db_name)
    # query = 'FOR doc IN SemanticStoryNodes_1 return doc'
    query = 'FOR doc IN StoryLine return doc'
    cursor = db.aql.execute(query, ttl=3600)
    graph_encoder = GraphEncoder()

    embedding_list = []
    meta_list = []

    cur_sentences = []
    last_movie = ''
    num_cur_stages = 0
    meta = {}
    max_sent_num = 0
    for cnt, sent_array in enumerate(cursor):
        # if sent_array['arango_id'] != 'Movies/92349435' and last_movie != 'Movies/92349435':
        #     continue
        # if sent_array['arango_movie'] != last_movie and last_movie != '':
        if sent_array['arango_id'] != last_movie and last_movie != '':
            final_sent = graph_encoder.process_scene_sentences(cur_sentences)

            # Previous method
            # final_sent = process_single_movie(cur_sentences, num_cur_stages + 1, proc_type)
            if len(final_sent) > max_sent_num:
                max_sent_num = len(final_sent)
            cur_sentences = []
            num_cur_stages = 0

            if len(final_sent) > 0:
                # embedding = graph_encoder.simple_graph_encoding(final_sent)
                embedding = graph_encoder.encode_with_transformers(final_sent)
                embedding_list.append(embedding.tolist()[0])
                # embedding_list.append(embedding)
                meta['sentence'] = final_sent
                meta_list.append(meta)

        cur_sentences.append(sent_array)
        num_cur_stages = max(num_cur_stages, sent_array['scene_element'])
        # print(sent_array)
        last_movie = sent_array['arango_id']
        meta = {
            'filename': 'none',
            'movie_id': sent_array['arango_id'],
            'nebula_movie_id': sent_array['arango_id'],
            'stage': sent_array['scene_element'],
            'frame_number': 'none',
            'sentence': sent_array['sentences'],
        }
        if cnt % 200 == 0:
            print(cnt)
        pass

    # final_sent = process_single_movie(cur_sentences, num_cur_stages + 1, proc_type)
    final_sent = graph_encoder.process_scene_sentences(cur_sentences)
    # embedding = graph_encoder.simple_graph_encoding(final_sent)
    if final_sent is not None:
        embedding = graph_encoder.encode_with_transformers(final_sent)
        embedding_list.append(embedding.tolist()[0])
        # embedding_list.append(embedding)
        meta['sentence'] = final_sent
        meta_list.append(meta)

    return embedding_list, meta_list


def load_vert_data():
    bert_data = '/home/migakol/data/bert_emb.pickle'
    file_handler = open(bert_data, 'rb')
    embedding_list_full = pickle.load(file_handler)
    file_handler.close()

    bert_data = '/home/migakol/data/bert_meta.pickle'
    file_handler = open(bert_data, 'rb')
    meta_list_full = pickle.load(file_handler)
    file_handler.close()

    bert_data = '/home/migakol/data/bert_emb_one.pickle'
    file_handler = open(bert_data, 'rb')
    embedding_list_one = pickle.load(file_handler)
    file_handler.close()

    bert_data = '/home/migakol/data/bert_meta_one.pickle'
    file_handler = open(bert_data, 'rb')
    meta_list_one = pickle.load(file_handler)
    file_handler.close()

    return embedding_list_full, meta_list_full, embedding_list_one, meta_list_one


def get_best_res_from_bert(bert_milvus, embedding_list):
    search_domain = bert_milvus.search_vector(3, embedding_list)
    # For loop is needed if we have two sentences or more
    for v in search_domain:
        print(v)


def get_similar_from_bert(graph_encoder, bert_milvus, sent):
    embedding = graph_encoder.simple_graph_encoding(sent)
    embedding = preprocessing.normalize(embedding.reshape(1, -1))

    get_best_res_from_bert(bert_milvus, embedding.tolist()[0])


def get_similar_from_matrix(embedding_full, graph_encoder, sent):
    embedding = graph_encoder.simple_graph_encoding(sent)
    embedding = preprocessing.normalize(embedding.reshape(1, -1))

    res = np.matmul(embedding_full, embedding.T)
    return np.argmax(res)



def check_bert_embeddings_locally():
    graph_encoder = GraphEncoder()
    bert_milvus = MilvusAPI('milvus', 'bert_embeddings', 'nebula_dev', 768)
    embedding_list_full, meta_list_full, embedding_list_one, meta_list_one = load_vert_data()
    # embedding into np.arraay
    embedding_full = np.array(embedding_list_full)
    embedding_full = preprocessing.normalize(embedding_full)

    # example = 'car and then blue lights'
    # get_similar_from_bert(graph_encoder, bert_milvus, example)

    get_best_res_from_bert(bert_milvus, embedding_list_full[0])


    pass

def save_bert_embeddings_locally():
    embedding_list, meta_list = get_sentences_from_movies(proc_type='separate')
    bert_data = '/home/migakol/data/bert_emb.pickle'
    file_handler = open(bert_data,  'wb')
    pickle.dump(embedding_list, file_handler)
    file_handler.close()

    bert_data = '/home/migakol/data/bert_meta.pickle'
    file_handler = open(bert_data, 'wb')
    pickle.dump(meta_list, file_handler)
    file_handler.close()

    embedding_list, meta_list = get_sentences_from_movies(proc_type='one')
    bert_data = '/home/migakol/data/bert_emb_one.pickle'
    file_handler = open(bert_data, 'wb')
    pickle.dump(embedding_list, file_handler)
    file_handler.close()

    bert_data = '/home/migakol/data/bert_meta_one.pickle'
    file_handler = open(bert_data, 'wb')
    pickle.dump(meta_list, file_handler)
    file_handler.close()


def movie2sentences():
    # bert_embeddings = MilvusAPI('milvus', 'bert_embeddings', 'nebula_dev', 768)
    # bert_embeddings = MilvusAPI('milvus', 'bert_embeddings', 'nebula_datadriven', 768)
    # bert_embeddings.drop_database()
    embedding_list, meta_list = get_sentences_from_movies('nebula_datadriven', proc_type='separate')



    # Put into
    bert_embeddings = MilvusAPI('milvus', 'bert_embeddings', 'nebula_datadriven', 768)
    bert_embeddings.insert_vectors(embedding_list, meta_list)

    print('Inserted data2')

#     #print("Movie: ", m)
#     centenses = ""
#     for i, sent_arry in enumerate(cursor_r):
#         print("Movie: ", m ," Stage: ", i)
#         if len(sent_arry.split("\"")) > 6: //6 means just "a lot" :)))
#             if sent_arry.split("\"")[1] != sent_arry.split("\"")[3]:// You can add more than 2 sentences, [5],[7].....
#                 print(sent_arry.split("\"")[1] + "." + sent_arry.split("\"")[3])
#                 centenses = centenses + "." + sent_arry.split("\"")[1] + "." + sent_arry.split("\"")[3]
#         elif len(sent_arry.split("\"")) > 1:
#             print(sent_arry.split("\"")[1])
#             centenses = centenses + "." + sent_arry.split("\"")[1]
#         # for t in sent_arry.split("\""):
#         #     print (t)
#     return centenses



def select_sentences():
    # sentences = MilvusAPI('milvus', 'scene_graph', 'nebula_dev', 640)
    # db = connect_db('nebula_dev')
    # conn = nebula_connect('nebula_dev')
    # adapter = Nebula_Networkx_Adapter(conn=conn)
    # all_movies = get_all_movies(db)
    # movie_num = 0

    movie2sentences()
    # save_bert_embeddings_locally()
    # check_bert_embeddings_locally()

    # query_r = 'FOR doc IN SemanticStoryNodes FILTER doc.arango_movie == "{}"  ' \
    #           'AND doc.description == \'With\' RETURN doc.sentence'.format(m)
    # cursor_r = db.aql.execute(query_r)
    # for cent_arry in cursor_r:
    #     for t in np.array(cent_arry, ): //
    #         Stage
    #     print(t)

    print('Done')


def select_triplets():
    db = connect_db('nebula_dev')
    query = 'FOR doc in milvus_scene_graph_visual_genome return doc'
    cursor = db.aql.execute(query, ttl=3600)

    triplet_arr = []

    for cnt, triplet in enumerate(cursor):
        triplet_arr.append(triplet['stage'])

    triplet_data = '/home/migakol/data/triplets.pickle'
    file_handler = open(triplet_data, 'wb')
    pickle.dump(triplet_arr, file_handler)
    file_handler.close()

def build_triplet_gt_separator():
    triplet_data = '/home/migakol/data/triplets.pickle'
    file_handler = open(triplet_data, 'rb')
    triplet_arr = pickle.load(file_handler)
    file_handler.close()

    random.seed(123)

    encoder = GraphEncoder()
    res = 0
    real_cnt = 0
    for k in range(200):
        v1 = random.randint(0, len(triplet_arr))
        v2 = random.randint(0, len(triplet_arr))
        sent1 = [triplet_arr[v1][0] + ' ' + triplet_arr[v1][1] + ' ' + triplet_arr[v1][2]]
        sent1.append(triplet_arr[v2][0] + ' ' + triplet_arr[v2][1] + ' ' + triplet_arr[v2][2])
        sent2 = [triplet_arr[v2][0] + ' ' + triplet_arr[v2][1] + ' ' + triplet_arr[v2][2]]
        sent2.append(triplet_arr[v1][0] + ' ' + triplet_arr[v1][1] + ' ' + triplet_arr[v1][2])

        if v1 == v2:
            continue

        real_cnt = real_cnt + 1
        emb1 = encoder.encode_with_transformers(sent1)
        emb2 = encoder.encode_with_transformers(sent2)

        emb1 = np.reshape(emb1, (1, 768)) / np.linalg.norm(emb1)
        emb2 = np.reshape(emb2, (768, 1)) / np.linalg.norm(emb2)
        res = res + np.matmul(emb1, emb2)
        pass

    print('Average res ', res / real_cnt, ' for ', real_cnt, ' tests')

def build_triplet_gt():
    triplet_data = '/home/migakol/data/triplets.pickle'
    file_handler = open(triplet_data, 'rb')
    triplet_arr = pickle.load(file_handler)
    file_handler.close()

    random.seed(123)

    encoder = GraphEncoder()
    res = 0
    real_cnt = 0
    for k in range(200):
        v1 = random.randint(0, len(triplet_arr))
        v2 = random.randint(0, len(triplet_arr))
        sent1 = triplet_arr[v1][0] + ' ' + triplet_arr[v1][1] + ' ' + triplet_arr[v1][2] + ' . ' + \
                triplet_arr[v2][0] + ' ' + triplet_arr[v2][1] + ' ' + triplet_arr[v2][2]
        sent2 = triplet_arr[v2][0] + ' ' + triplet_arr[v2][1] + ' ' + triplet_arr[v2][2] + ' . ' + \
                triplet_arr[v1][0] + ' ' + triplet_arr[v1][1] + ' ' + triplet_arr[v1][2]

        if v1 == v2:
            continue

        real_cnt = real_cnt + 1
        emb1 = encoder.simple_graph_encoding(sent1)
        emb2 = encoder.simple_graph_encoding(sent2)

        emb11 = encoder.encode_with_transformers(sent1)
        emb22 = encoder.encode_with_transformers(sent2)

        emb1 = np.reshape(emb1, (1, 768)) / np.linalg.norm(emb1)
        emb2 = np.reshape(emb2, (768, 1)) / np.linalg.norm(emb2)
        res = res + np.matmul(emb1, emb2)
        pass

    print('Average res ', res / real_cnt, ' for ', real_cnt, ' tests')

def debug_bert():
    # connect = DatabaseConnector()

    bert_embeddings = MilvusAPI('milvus', 'bert_embeddings', 'nebula_dev', 768)
    db = connect_db('nebula_datadriven')

    query = 'FOR doc in milvus_bert_embeddings return doc'
    query = 'FOR doc IN StoryLine return doc'
    cursor = db.aql.execute(query, ttl=3600)

    val_array = []
    for data in cursor:
        val = int(data['arango_id'][7:])
        val_array.append(val)
        # print(data)

    # encoder = GraphEncoder()
    # index = MilvusAPI('milvus', 'bert_embeddings', 'nebula_dev', 768)
    #
    # query = 'FOR doc IN nebula_bert_embeddings FILTER doc.nebula_movie_id == "{}"  RETURN doc'.format(movie_id)
    # cursor = db.aql.execute(query)
    # vectors = []
    # for data in cursor:
    #     # print(data)
    #     vectors.append(int(data['milvus_key']))
    # return (vectors)

    pass


if __name__ == '__main__':
    print('Creating BERT embeddings')
    # debug_bert()
    # 92453176
    encode_movie_to_bert('Movies/92349435', 'nebula_datadriven')
    select_sentences()
    # select_triplets()
    # build_triplet_gt()
    # build_triplet_gt_separator()
