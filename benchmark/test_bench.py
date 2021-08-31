from sys import prefix
from gensim.models.doc2vec import TaggedDocument
from benchmark.input_options import TrainOptions
from benchmark.nlp_benchmark import NebulaStoryEvaluation
from benchmark.clip_benchmark import NebulaVideoEvaluation
from benchmark.connection import connect_db
from benchmark.connection import get_all_movies
from benchmark.connection import get_all_stories

from multiprocessing.managers import all_methods
import networkx as nx

from numpy.lib.function_base import append, average
from scipy.ndimage.measurements import label
from scipy.stats.stats import mode
from embeddings.nebula_networkx_adapter import Nebula_Networkx_Adapter
from arango import ArangoClient

import pickle
import numpy as np

import sys

from milvus_api.milvus_api import MilvusAPI
from milvus import Milvus

import logging
import os
import pandas as pd


# def connect_db(dbname):
#     client = ArangoClient(hosts='http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:8529')
#     # client = ArangoClient(hosts='http://18.159.140.240:8529')
#     db = client.db(dbname, username='nebula', password='nebula')
#     return (db)
#
# def nebula_connect(db_name: str) -> dict:
#     conn = {'dbName': db_name,
#            'username': 'nebula',
#            'password': 'nebula',
#            'hostname': '18.159.140.240',
#            'protocol': 'http',
#            'port': 8529}
#     return conn

def nebula_get_graph_formdb(ma, _filter):
    attributes = { 'vertexCollections':
                                    {'Actors': {'labels','description'},'Actions': {'labels','description'},
                                    'Relations': {'labels','description'}, 'Properties':{'labels','description'}},
                    'edgeCollections' :
                                    {'ActorToAction': {'_from', '_to','labels'},'ActorToRelation': {'_from', '_to','labels'},
                                    'MovieToActors':{'_from', '_to', 'labels'}, 'RelationToProperty':{'_from', '_to', 'labels'}}}

    # Export networkX graph
    _filter = _filter
    g, lables, descriptions = ma.create_nebula_graph(graph_name = 'Test',  graph_attributes = attributes, _filter = _filter)
    fitG = nx.convert_node_labels_to_integers(g, first_label=0)
    return fitG, lables, descriptions


def nebula_get_stories(all_movies, ma):
    story = 0
    documents = []
    tags = {}
    nebula_metadata = {}
    for movie in all_movies.values():
        fitG, lables, descriptions = nebula_get_graph_formdb(ma, movie['movie']['movie_id'])
        prefix_labels = []
        stories = []
        successors = list(nx.dfs_preorder_nodes(fitG))

        for successor in successors:
            attrebutes = ""
            # print(fitG.nodes[successor]['attr_dict']['_class'], successor)
            nebs = dict(nx.bfs_successors(fitG, successor))
            # print(nebs)
            if lables[successor]:
                if int(lables[successor][0]) > 0 and ((int(lables[successor][2]) - int(lables[successor][1])) > 3):
                    if fitG.nodes[successor]['attr_dict']['_class'] == "person" or fitG.nodes[successor]['attr_dict'][
                        '_class'] == "car":
                        stories.append(fitG.nodes[successor]['attr_dict']['_object'])
                        stories.append(
                            fitG.nodes[successor]['attr_dict']['_object'] + "_from_" + lables[successor][1] + "_to_" +
                            lables[successor][2])

            if (len(nebs) > 2):
                for neb in nebs[successor]:
                    if neb in nebs:
                        for next_neb in nebs[neb]:
                            if (fitG.nodes[successor]['attr_dict']['_class'] != fitG.nodes[next_neb]['attr_dict'][
                                '_class']):
                                _prefix = fitG.nodes[successor]['attr_dict']['_class'].replace(" ", "_").replace("(",
                                                                                                                 '').replace(
                                    ")", '').replace(".", '').replace(",", '')
                                _base = fitG.nodes[neb]['attr_dict']['_class'].replace(" ", "_").replace("(",
                                                                                                         '').replace(
                                    ")", '').replace(".", '').replace(",", '')
                                _suffix = fitG.nodes[next_neb]['attr_dict']['_class'].replace(" ", "_").replace("(",
                                                                                                                '').replace(
                                    ")", '').replace(".", '').replace(",", '')
                                sentence = _prefix + "_" + _base + "_" + _suffix
                                if ("Then" not in sentence) or ("With" not in sentence):
                                    # stories.append(sentence)
                                    stories.append(_prefix)
                                    stories.append(_base)
                                    stories.append(_suffix)
            _tag = "story_" + str(story)
            dfs_doc = TaggedDocument(words=stories, tags=[_tag])

        # print(movie['movie']['_id'])
        # print("DFS-based neibs")
        # print(_tag)
        # print (len(stories))
        # print(dfs_doc)
        # input("Press Enter to continue...")
        documents.append(dfs_doc)
        tags[story] = _tag
        nebula_metadata[story] = (movie['movie']['file_name'], movie['movie']['_id'])
        print(nebula_metadata[story], story)
        story = story + 1

    print("Number of stories:", story)
    # print(documents)
    return (documents, tags, nebula_metadata)


def save_data(db_name: str, stories_out: str):
    """
    Save strings generated by Nebula graph extraction
    :param db_name: the name of the database that we connect to
    :param stories_out: the name of the file where we save the pickle with stories
    :return:
    """
    print('Saving stories')
    conn = nebula_connect(db_name)
    adapter = Nebula_Networkx_Adapter(conn=conn)
    db = connect_db(db_name)

    if db.has_collection('Stories'):
        print('Stories')
        # db.delete_collection('Stories')
    # db.create_collection('Stories')
    all_movies = adapter.nebula_get_all_movies()
    stories, _tags, nebula_meta = nebula_get_stories(all_movies, adapter)
    print('Ended creating stories')

    file_handle = open(stories_out, 'wb')
    pickle.dump(stories, file_handle)


def build_nlp_benchmarks(db_name: str, stories_in: str, benchmark_out: str):
    """
    Build benchmark based on NLP
    :param db_name: the name of the database that we connect to
    :param stories_in: the input file with the stories built by Nebula graph detection
    :param benchmark_out: the name of the pickle file where we will save the benchmarks
    :return:
    """
    with open(stories_in, 'rb') as f:
        stories = pickle.load(f)
        story_benchmark = NebulaStoryEvaluation()
        story_benchmark.build_similarity_matrices(stories)

    pass


def create_embeddings_from_files():
    data_folder = '/home/migakol/data'
    list_of_all_embeddings = []
    # Go over all embeddings
    for k in range(1859):
        file_name = os.path.join(data_folder, 'embedding_folder', 'embedding' + f'{k:04d}' + '.pickle')
        file_handle = open(file_name, 'rb')
        embedding = pickle.load(file_handle)
        list_of_all_embeddings.append(embedding)
        file_handle.close()

    file_handle = open(os.path.join(data_folder, 'all_embeddings.pickle'), 'wb')
    pickle.dump(list_of_all_embeddings, file_handle)
    file_handle.close()



def save_embeddings_to_milvus(all_movies, list_of_all_embeddings):
    data_folder = '/home/migakol/data'

    clip_embeddings = MilvusAPI('milvus', 'clip_string_embeddings', 'nebula_dev', 640)
    meta_list = []

    for k in range(len(all_movies)):
        # if k < 1581:
        #     continue
        log_file = open(os.path.join(data_folder, 'logbench.log'), 'w')
        print('Creating meta for movie ', k, file=log_file)
        log_file.close()

        meta = {
            'filename': 'none',
            'movie_id': all_movies[k]['movie']['_id'],
            'frame_start': all_movies[k]['movie']['split'],
            'sentence': all_movies[k]['movie']['splits_total'],
        }

        meta_list


def create_embeddings_3(clip_bench, all_movies, thresholds):
    pass


def create_embeddings_2(clip_bench, all_movies, thresholds):
    """
    First, we create embeddings for all movies
    :return:
    """
    data_folder = '/home/migakol/data'

    list_of_all_embeddings = []
    for k in range(len(all_movies)):
        # if k < 1581:
        #     continue
        log_file = open(os.path.join(data_folder, 'logbench.log'), 'w')
        print('Creating embedding for movie ', k, file=log_file)
        log_file.close()
        movie_name = '/movies/' + all_movies[k]['movie']['file_name'] + '.avi'
        start_time = float(all_movies[k]['movie']['split'])
        end_time = float(all_movies[k]['movie']['splits_total'])
        key = all_movies[k]['movie']['_key']

        # movie_name = all_movies[k]['movie']['full_path']

        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds,
                                                                           start_time=start_time, end_time=end_time)
        list_of_all_embeddings.append(embedding_list)

        # file_name = os.path.join(data_folder, 'embedding_folder', 'embedding' + f'{k:04d}' + '.pickle')
        file_name = os.path.join(data_folder, 'embedding_folder', 'embedding_' + key + '.pickle')
        file_handle = open(file_name, 'wb')
        pickle.dump(embedding_list, file_handle)

    file_handle = open(os.path.join(data_folder, 'all_embeddings.pickle'), 'wb')
    pickle.dump(list_of_all_embeddings, file_handle)
    return list_of_all_embeddings


def create_embeddings(clip_bench, all_movies, thresholds):
    """
    First, we create embeddings for all movies
    :return:
    """
    data_folder = '/home/migakol/data'

    # file_handle = open(os.path.join(data_folder, 'all_embeddings.pickle'), 'rb')
    # list_of_all_embeddings = pickle.load(file_handle)
    # return list_of_all_embeddings

    list_of_all_embeddings = []
    for k in range(len(all_movies)):
        log_file = open(os.path.join(data_folder, 'logbench.log'), 'w')
        print('Creating embedding for movie ', k, file=log_file)
        log_file.close()
        movie_name = '/movies/' + all_movies[k]['movie']['file_name'] + '.avi'
        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds)
        list_of_all_embeddings.append(embedding_list)

        file_name = os.path.join(data_folder, 'embedding_folder', 'embedding' + f'{k:04d}' + '.pickle')
        file_handle = open(file_name, 'wb')
        pickle.dump(embedding_list, file_handle)

    file_handle = open(os.path.join(data_folder, 'all_embeddings.pickle'), 'wb')
    pickle.dump(list_of_all_embeddings, file_handle)
    return list_of_all_embeddings


def compute_similarity_matrix(all_movies, list_of_all_embeddings, clip_bench):
    """
    :param all_movies: as returned b adapter.nebula_get_all_movies()
    :param list_of_all_embeddings: list of list of array.
            list_of_all_embeddings[k] - movie k embeddings
            list_of_all_embeddings[k][j] - for threshold j, Nx512, where N is the number of detected scenes
    :param clip_bench:
    :return:
    """
    similarity_matrix = np.zeros((len(all_movies), len(all_movies)))

    logging.basicConfig(filename='/home/migakol/data/bench.log', level=logging.DEBUG)
    for k in range(len(all_movies) - 1):

        data_folder = '/home/migakol/data'
        log_file = open(os.path.join(data_folder, 'logbench.log'), 'w')
        print('Creating similarity for movie ', k, file=log_file)
        log_file.close()
        print('Testing movie ', k)
        # movie_name1 = '/movies/' + all_movies[k]['movie']['file_name'] + '.avi'
        # embedding_list1 = clip_bench.create_clip_representation(movie_name1, thresholds=thresholds)
        embedding_list1 = list_of_all_embeddings[k]
        if len(embedding_list1) == 0:
            print('Movie is bad ', k)
            continue
        if len(embedding_list1[0]) == 0 or len(embedding_list1[1]) == 0 or len(embedding_list1[2]) == 0 or len(
                embedding_list1[3]) == 0:
            continue
        for m in range(k + 1, len(all_movies)):
            # movie_name2 = '/movies/' + all_movies[m]['movie']['file_name'] + '.avi'
            # embedding_list2 = clip_bench.create_clip_representation(movie_name2, thresholds=thresholds)
            embedding_list2 = list_of_all_embeddings[m]
            if len(embedding_list2) == 0 or len(embedding_list2[0]) == 0:
                print('Second Movie is bad ', m)
                continue
            if len(embedding_list2[0]) == 0 or len(embedding_list2[1]) == 0 or len(embedding_list2[2]) == 0 or len(
                    embedding_list2[3]) == 0:
                continue
            sim = clip_bench.find_similarity(embedding_list1, embedding_list2)
            similarity_matrix[k, m] = sim
            logging.info('Pair ' + str(k) + " " + str(m))

    # thresholds = [6.6, 7.0, 7.5]
    # movie_name = '/movies/' + all_movies[1]['movie']['file_name'] + '.avi'

    f = open('/home/migakol/data/sim_benchmark.npy', 'wb')
    np.save(f, similarity_matrix)

    return similarity_matrix


def build_clip_benchmark_3(db_name):
    clip_bench = NebulaVideoEvaluation()
    db = connect_db(db_name)
    query = 'FOR doc IN Movies RETURN doc'
    # query = 'FOR doc IN milvus_bert_embeddings RETURN doc'
    cursor = db.aql.execute(query, ttl=3600)
    thresholds = [0.6, 0.7, 0.8]
    list_of_all_embeddings = []
    data_folder = '/home/migakol/data'

    # Create a list of movie names
    name_arr = []
    for k, data in enumerate(cursor):
        name_arr.append('/movies/' + data['movie_name'] + '.avi')
        # name_arr.append(data['movie_id'])

    for k, movie_name in enumerate(name_arr):
        log_file = open(os.path.join(data_folder, 'logbench.log'), 'w')
        print('Creating similarity for movie ', k, file=log_file)
        # movie_name = '/movies/' + data['movie_name'] + '.avi'
        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds)
        list_of_all_embeddings.append(embedding_list)

    file_handle = open(os.path.join(data_folder, 'all_embeddings.pickle'), 'wb')
    pickle.dump(list_of_all_embeddings, file_handle)

    similarity_matrix = np.zeros((len(list_of_all_embeddings), len(list_of_all_embeddings)))

    for k in range(len(list_of_all_embeddings) - 1):
        log_file = open(os.path.join(data_folder, 'logbench.log'), 'w')
        print('Computing similarity for movie ', k, file=log_file)
        log_file.close()
        print('Testing movie ', k)
        embedding_list1 = list_of_all_embeddings[k]
        if len(embedding_list1) == 0:
            print('Movie is bad ', k)
            continue
        if len(embedding_list1[0]) == 0:
            continue
        for m in range(k + 1, len(list_of_all_embeddings)):
            embedding_list2 = list_of_all_embeddings[m]
            if len(embedding_list2) == 0 or len(embedding_list2[0]) == 0:
                print('Second Movie is bad ', m)
                continue
            sim = clip_bench.find_similarity(embedding_list1, embedding_list2)
            similarity_matrix[k, m] = sim
            logging.info('Pair ' + str(k) + " " + str(m))

    # thresholds = [6.6, 7.0, 7.5]
    # movie_name = '/movies/' + all_movies[1]['movie']['file_name'] + '.avi'

    f = open('/home/migakol/data/sim_benchmark.npy', 'wb')
    np.save(f, similarity_matrix)

def build_clip_benchmark_2(db_name):
    print('Saving stories')
    # conn = nebula_connect(db_name)
    # adapter = Nebula_Networkx_Adapter(conn=conn)
    db = connect_db(db_name)

    all_movies = get_all_movies(db)
    all_movies = get_all_stories(db)
    clip_bench = NebulaVideoEvaluation()

    # for m in range(len(all_movies)):
    #     if all_movies[m]['movie']['_key'] == '15615745':
    #         print(m)

    # thresholds = [6.3, 6.8, 7.3, 7.8]
    # thresholds = [6.5, 7.0, 7.5, 8.5]
    thresholds = [0.5, 0.6, 0.7, 0.8]
    thresholds = [0.7]

    list_of_all_embeddings = create_embeddings_2(clip_bench, all_movies, thresholds)
    # return

    # data_folder = '/home/migakol/data'
    # file_handle = open(os.path.join(data_folder, 'all_embeddings.pickle'), 'rb')
    # list_of_all_embeddings = pickle.load(file_handle)

    # save_embeddings_to_milvus(all_movies, list_of_all_embeddings)

    # data_folder = '/home/migakol/data/embedding_folder'
    # list_of_all_embeddings = []
    # onlyfiles = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    # for f in onlyfiles:
    #     file_name = os.path.join(data_folder, f)
    #     file_handle = open(file_name, 'rb')
    #     embedding_list = pickle.load(file_handle)
    #     list_of_all_embeddings.append(embedding_list)

    similarity_matrix = compute_similarity_matrix(all_movies, list_of_all_embeddings, clip_bench)



def build_clip_benchmark(db_name):
    print('Saving stories')
    conn = nebula_connect(db_name)
    adapter = Nebula_Networkx_Adapter(conn=conn)
    db = connect_db(db_name)

    all_movies = adapter.nebula_get_all_movies()

    if db.has_collection('Stories'):
        print('Stories')
        # db.delete_collection('Stories')
    # db.create_collection('Stories')
    # all_movies = adapter.nebula_get_all_movies()

    # stories, _tags, nebula_meta = nebula_get_stories(all_movies, adapter)
    similarity_matrix = np.zeros((len(all_movies), len(all_movies)))

    clip_bench = NebulaVideoEvaluation()

    # thresholds = [6.3, 6.8, 7.3, 7.8]
    thresholds = [6.5, 7.0, 7.5]
    # video_eval = NebulaVideoEvaluation()

    list_of_all_embeddings = create_embeddings(clip_bench, all_movies, thresholds)
    similarity_matrix = compute_similarity_matrix(all_movies, list_of_all_embeddings, clip_bench)

    # logging.basicConfig(filename='/home/migakol/data/bench.log', level=logging.DEBUG)
    # for k in range(len(all_movies) - 1):
    #
    #     data_folder = '/home/migakol/data'
    #     log_file = open(os.path.join(data_folder, 'logbench.log'), 'w')
    #     print('Creating distances for movie ', k, file=log_file)
    #     log_file.close()
    #     print('Testing movie ', k)
    #     # movie_name1 = '/movies/' + all_movies[k]['movie']['file_name'] + '.avi'
    #     # embedding_list1 = clip_bench.create_clip_representation(movie_name1, thresholds=thresholds)
    #     embedding_list1 = list_of_all_embeddings[k]
    #     if len(embedding_list1) == 0:
    #         print('Movie is bad ', k)
    #         continue
    #     for m in range(k+1, len(all_movies)):0
    #         # movie_name2 = '/movies/' + all_movies[m]['movie']['file_name'] + '.avi'
    #         # embedding_list2 = clip_bench.create_clip_representation(movie_name2, thresholds=thresholds)
    #         embedding_list2 = list_of_all_embeddings[m]
    #         if len(embedding_list2) == 0:
    #             print('Second Movie is bad ', m)
    #             continue
    #         sim = clip_bench.find_similarity(embedding_list1, embedding_list2)
    #         similarity_matrix[k, m] = sim
    #         logging.info('Pair ' + str(k) + " " + str(m))
    #
    # # thresholds = [6.6, 7.0, 7.5]
    # # movie_name = '/movies/' + all_movies[1]['movie']['file_name'] + '.avi'
    #
    # f = open('/home/migakol/sim_benchmark.npy', 'wb')
    # np.save(f, similarity_matrix)

    pass


def test_sentence_arrango(debug_mode=False):

    host = '127.0.0.1'
    port = '19530'
    client = Milvus(host, port)
    # Connect to the Milvus database
    sentences = MilvusAPI('lables', 'nebula_dev')
    # Read all the files
    conn = nebula_connect('nebula_dev')
    adapter = Nebula_Networkx_Adapter(conn=conn)
    all_movies = adapter.nebula_get_all_movies()
    data_folder = '/home/migakol/data/embedding_folder'
    sentence_folder = '/home/migakol/data/sentences'
    onlyfiles = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    # Go over all movies.
    # Save the data for each movie. We save 3 different embeddings

    db = connect_db('nebula_dev')

    if not debug_mode:
        if db.has_collection('hollywood_clip_sentences'):
            db.delete_collection('hollywood_clip_sentences')
        sentences_collection = db.create_collection('hollywood_clip_sentences')
        sentences_collection.add_hash_index(fields=['_key'])
    else:
        sentences_collection = db.collection('hollywood_clip_sentences')

    # holly_clips = db.collection('hollywood_clip_sentences')

    num_zero_files = 0

    if debug_mode:
        for k in all_movies:
            if all_movies[k]['movie']['_key'] == '10291014':
                print(k)

    for cnt, file_name in enumerate(onlyfiles):
        if debug_mode:
            if file_name != 'embedding_10291014.pickle':
                continue
        single_movie_list = []
        with open(os.path.join(data_folder, file_name), 'rb') as file_handler:
            # Embeddings from all threhsolds
            all_threshold_embeddings = pickle.load(file_handler)
            for single_threshold_embedding in all_threshold_embeddings:
                single_threshold_list = []
                # Go over all key-frames (embeddings)
                for e_num in range(single_threshold_embedding.shape[0]):
                    embedding_as_list = single_threshold_embedding[e_num, :].tolist()
                    # Take one vector
                    search_domain = sentences.search_vector(1, embedding_as_list)
                    # For loop is needed if we have two sentences or more
                    for v in search_domain:
                        single_threshold_list.append(v[1]['sentence'])
                        # print(v)
                single_movie_list.append(single_threshold_list)
        # Save the movie list in pickle file
        dot_loc = file_name.find('.')
        # file_handle = open(os.path.join(sentence_folder, 'sentences_' + file_name[10:dot_loc] + '.pickle'), 'wb')
        # pickle.dump(single_movie_list, file_handle)
        print(cnt)
        if not debug_mode:
            if len(single_movie_list) != 4:
                sentences_collection.insert(
                    {'_key': file_name[10:dot_loc], 'th0': [], 'th1': [], 'th2': [], 'th3': []})
            else:
                sentences_collection.insert(
                    {'_key': file_name[10:dot_loc], 'th0': single_movie_list[0], 'th1': single_movie_list[1],
                    'th2': single_movie_list[2], 'th3': single_movie_list[3]})
        else:
            if len(single_movie_list) != 4:
                num_zero_files = num_zero_files + 1

    log_file = open(os.path.join(data_folder, 'logbench.log'), 'w')
    print('Zero files ', num_zero_files, file=log_file)
    log_file.close()


    pass


def read_clip_benchmark(db_name):
    db = connect_db('nebula_dev')

    holly_clip = db.collection('hollywood_clip_sentences')
    cursor = db.aql.execute('FOR doc IN hollywood_clip_sentences FILTER doc._key == 12096264 RETURN doc')
    student_names = [document for document in cursor]

    benchmark_collection = db.collection('Benchmark')
    cursor = db.aql.execute('FOR doc IN Benchmark RETURN doc')

    # if db.has_collection('students'):
    #     students = db.collection('students')
    # else:
    #     students = db.create_collection('students')
    #
    # students.insert({'name': 'jane', 'age': 19})
    # students.insert({'name': 'josh', 'age': 18})
    # students.insert({'name': 'jake', 'age': 21})
    # Execute an AQL query. This returns a result cursor.
    cursor = db.aql.execute('FOR doc IN Benchmark RETURN doc')
    # Iterate through the cursor to retrieve the documents.
    student_names = [document['index1'] for document in cursor]

    pass


def write_clip_benchmark_into_arrango(db_name):

    # Create a new collection named "benchmark".
    db = connect_db('nebula_dev')
    if db.has_collection('benchmark'):
        db.delete_collection('benchmark')
    benchmark_collection = db.create_collection('benchmark')

    # conn = nebula_connect('nebula_dev')
    # adapter = Nebula_Networkx_Adapter(conn=conn)
    all_movies = get_all_movies(db)

    # load matrix with embeddings
    similarity_filename = '/home/migakol/data/sim_benchmark.npy'
    similarity_matrix = np.load(similarity_filename)

    # Go over all movie pair
    for k in range(len(all_movies) - 1):
        for m in range(k + 1, len(all_movies)):
            benchmark_collection.insert({'index1': k, 'index2': m, 'sim_value': similarity_matrix[k, m]})

    pass

    # key1, key2, similarity


def debug_movie(movie_key):
    conn = nebula_connect('nebula_dev')
    adapter = Nebula_Networkx_Adapter(conn=conn)
    all_movies = adapter.nebula_get_all_movies()
    data_folder = '/home/migakol/data'
    clip_bench = NebulaVideoEvaluation()
    thresholds = [6.5, 7.0, 7.5, 8.5]

    sentences = MilvusAPI('lables', 'nebula_dev')

    for k in range(len(all_movies)):
        if all_movies[k]['movie']['_key'] != movie_key:
            continue
        log_file = open(os.path.join(data_folder, 'logbench.log'), 'w')
        print('Creating embedding for movie ', k, file=log_file)
        log_file.close()
        movie_name = '/movies/' + all_movies[k]['movie']['file_name'] + '.avi'
        start_time = float(all_movies[k]['movie']['split'])
        end_time = float(all_movies[k]['movie']['splits_total'])
        key = all_movies[k]['movie']['_key']
        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds,
                                                                           start_time=start_time, end_time=end_time)
        print(1)

        single_threshold_list = []
        # Go over all key-frames (embeddings)
        for e_num in range(embedding_list[3].shape[0]):
            embedding_as_list = embedding_list[0][e_num, :].tolist()
            # Take one vector
            search_domain = sentences.search_vector(1, embedding_as_list)
            # For loop is needed if we have two sentences or more
            for v in search_domain:
                single_threshold_list.append(v[1]['sentence'])
                # print(v)
        print(1)
        # embedding_as_list = single_threshold_embedding[e_num, :].tolist()
        # # Take one vector
        # search_domain = sentences.search_vector(1, embedding_as_list)

    print('Done')


def check_dima():
    db = connect_db('nebula_dev')
    query = 'RETURN LENGTH(milvus_scene_graph)'  # get all vectors from arango
    cursor = db.aql.execute(query)
    len_milvus = 0
    for data in cursor:
        len_milvus = data

    cnt = 0
    add_cnt = 2000
    id = 0
    while cnt < len_milvus:
        query = f'FOR doc IN milvus_scene_graph SORT doc._key LIMIT {cnt}, {add_cnt} RETURN doc'
        cursor = db.aql.execute(query)
        embeddings = np.zeros((add_cnt, 640))
        sentences = []
        k = 0
        for data in cursor:
            sentences.append(data['stage'])
            embeddings[k, :] = np.array(data['frame_number'])
            k = k + 1
        cnt = cnt + add_cnt
        f = open(f'/home/migakol/data/milvus/embedding{id:03d}.npy', 'wb')
        np.save(f, embeddings)
        f = open(f'/home/migakol/data/milvus/sentences{id:03d}.npy', 'wb')
        pickle.dump(sentences, f)
        id = id + 1

    print(1)

    # for

    """
    SORT u.firstName, u.lastName, u.id DESC
  LIMIT 2, 5
  """

def create_similarity_matrix(db_name):
    """
    Given the embeddings of all movies
    :return:
    """
    # f = open('/home/migakol/data/sim_benchmark.npy', 'rb')
    # embeddings = np.load('/home/migakol/data/sim_benchmark.npy')

    db = connect_db(db_name)
    all_movies = get_all_movies(db)

    # Create list of strings from names
    column_names = ["_id", "split", "splits_total"]
    df = pd.DataFrame(columns=column_names)
    for key in all_movies.keys():
        row = {'_id': all_movies[key]['movie']['_id'], 'split': int(all_movies[key]['movie']['split']),
               'splits_total': int(all_movies[key]['movie']['splits_total'])}
        df = df.append(row, ignore_index=True)
        pass

    df.to_pickle("/home/migakol/data/sim_data.pkl")
    pass

    # Go over all pair

if __name__ == '__main__':

    # read_clip_benchmark('')

    # tt = '/home/migakol/data/sentences/embedding0531.pickle'
    # file_handle = open(tt, 'rb')
    # ff = pickle.load(file_handle)
    # test_sentence_arrango(debug_mode=True)
    # 15615745
    # debug_movie('10291014')
    # check_dima()
    # create_embeddings_from_files()

    build_clip_benchmark_3('nebula_datadriven')
    # build_clip_benchmark_2('nebula_datadriven')

    # opt = TrainOptions().parse()
    # if opt.mode == 'save_data':
    #     save_data(db_name=opt.db_name, stories_out=opt.stories_out)
    # elif opt.mode == 'nlp_benchmark':
    #     build_nlp_benchmarks(db_name=opt.db_name, stories_in=opt.stories_in, benchmark_out=opt.benchmark_out)
    # elif opt.mode == 'clip_benchmark':
    #     # get_embeddinngs_from_movies(db_name=opt.db_name, proc_type='full')
    #
    #     # build_clip_benchmark_2(db_name=opt.db_name)
    #     build_clip_benchmark_2('nebula_datadriven')
    #     # create_similarity_matrix(opt.db_name)
    #
    #     # write_clip_benchmark_into_arrango('')
    #     # test_sentence_arrango(debug_mode=False)

