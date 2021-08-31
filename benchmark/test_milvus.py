"""
This file ests the quality of milvus by creating a separate brute-force data set and comparing milvus results with it
"""
from benchmark.connection import connect_db
from benchmark.connection import get_all_movies
import numpy as np
import pickle
import os

from embeddings.nebula_networkx_adapter import Nebula_Networkx_Adapter
from benchmark.clip_benchmark import NebulaVideoEvaluation
from milvus_api.milvus_api import MilvusAPI


def save_milvus_data(single_file_size=2000):
    """
    :param single_file_size: the size of a single file
    :return:
    """
    db = connect_db('nebula_dev')
    query = 'RETURN LENGTH(milvus_scene_graph)'  # get all vectors from arango
    cursor = db.aql.execute(query)
    len_milvus = 0
    for data in cursor:
        len_milvus = data

    cnt = 0
    add_cnt = single_file_size
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
        f = open(f'/home/migakol/data/milvus/sentences{id:03d}.pkl', 'wb')
        pickle.dump(sentences, f)
        id = id + 1

    print(1)


def load_milvus_brute_force():
    """
    Load Milvus data saved in files
    :return:
    """
    num_files = 777
    data_folder = '/home/migakol/data/milvus'
    # Get the ID numbers of embedding and sentences files
    file_names = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]
    number_list = []
    for file_name in file_names:
        if file_name[0:3] == 'all':
            continue
        number_list.append(int(file_name[-7:-4]))

    number_list = list(set(number_list))

    embeddings = np.zeros((0, 640))
    sentences = []

    start_cnt = 0
    total_cnt = 0
    for k in number_list:
        embedding_name = os.path.join(data_folder, f'embedding{k:03d}.npy')
        sentence_name = os.path.join(data_folder, f'sentences{k:03d}.npy')

        cur_embedding = np.load(embedding_name)
        add_cnt = cur_embedding.shape[0]
        embeddings = np.append(embeddings, cur_embedding, axis=0)

        file_handle = open(sentence_name, 'rb')
        cur_sentences = pickle.load(file_handle)

        start_cnt = start_cnt + add_cnt
        sentences = [*sentences, *cur_sentences]

        total_cnt = total_cnt + 1

        if total_cnt % 20 == 0:
            print(total_cnt)

    f = open(f'/home/migakol/data/milvus/all_embeddings.npy', 'wb')
    np.save(f, embeddings)

    f = open(f'/home/migakol/data/milvus/all_sentences1.pkl', 'wb')
    pickle.dump(sentences, f)


    return embeddings, sentences


def test_milvus(movie_key):
    """
    Open a movie, find CLIP embeddings and check that the sentences returned from milvus are similar to what I get
    :return:
    """
    # conn = nebula_connect('nebula_dev')
    # adapter = Nebula_Networkx_Adapter(conn=conn)
    db = connect_db('nebula_dev')
    all_movies = get_all_movies(db)
    data_folder = '/home/migakol/data'
    clip_bench = NebulaVideoEvaluation()
    thresholds = [6.5, 7.0, 7.5, 8.5]

    # sentences = MilvusAPI('lables', 'nebula_dev')
    # sentences = MilvusAPI('scene_graph', 'nebula_dev')
    sentences = MilvusAPI('milvus', 'scene_graph', 'nebula_dev', 640)

    # presaved_embeddings = np.load('/home/migakol/data/milvus/all_embeddings.npy')
    #
    # f = open(f'/home/migakol/data/milvus/all_sentences.pkl', 'rb')
    # presaved_sentences = pickle.load(f)
    #
    # for k in range(presaved_embeddings.shape[0]):
    #     presaved_embeddings[k, :] = presaved_embeddings[k, :] / (np.linalg.norm(presaved_embeddings[k, :]) +
    #                                                              np.finfo(float).eps)

    all_coefs = []
    for k in range(len(all_movies)):
        # if k < 21:
        #     continue
        # if all_movies[k]['movie']['_key'] != movie_key:
        #     continue
        # log_file = open(os.path.join(data_folder, 'logbench.log'), 'w')
        # print('Creating embedding for movie ', k, file=log_file)
        # log_file.close()
        movie_name = '/movies/' + all_movies[k]['movie']['file_name'] + '.avi'
        start_time = float(all_movies[k]['movie']['split'])
        end_time = float(all_movies[k]['movie']['splits_total'])
        key = all_movies[k]['movie']['_key']
        # Get embeddings
        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds,
                                                                           start_time=start_time, end_time=end_time)

        # Get threshold according to Milvus
        single_threshold_list = []
        # Go over all key-frames (embeddings)
        for e_num in range(embedding_list[3].shape[0]):
            embedding_as_list = embedding_list[0][e_num, :].tolist()
            # Take one vector
            search_domain = sentences.search_vector(3, embedding_as_list)
            # For loop is needed if we have two sentences or more
            for v in search_domain:
                single_threshold_list.append(v[1]['sentence'])
                # print(v[0])
                # print(v[1]['sentence'])

            # Get sentences according to my method
            # emb_norm = embedding_list[0][e_num, :].T / np.linalg.norm(embedding_list[0][e_num, :])
            # dist = np.matmul(presaved_embeddings, emb_norm)
            # ind = np.argmax(dist)
            # # print(dist[ind], presaved_sentences[ind])
            # all_coefs.append(dist[ind])
            # if dist[ind] > 0.4:
            #     print(k, dist[ind], presaved_sentences[ind])
            # dist[ind] = 0
            # ind = np.argmax(dist)
            # # print(dist[ind], presaved_sentences[ind])
            # dist[ind] = 0
            # ind = np.argmax(dist)
            # print(dist[ind], presaved_sentences[ind])


    print(1)


if __name__ == '__main__':
    print('Started')
    # save_milvus_data(single_file_size=2000)
    # embeddings, sentences = load_milvus_brute_force()
    test_milvus('10291014')
    print('Done')
