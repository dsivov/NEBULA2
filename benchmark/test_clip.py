from benchmark.clip_benchmark import NebulaVideoEvaluation
from benchmark.connection import connect_db
from benchmark.connection import get_all_movies
from milvus_api.milvus_api import MilvusAPI
import numpy as np
import pickle
import os

import torch
import clip

import pandas as pd

from embeddings.nebula_embeddings_api import EmbeddingsLoader


def test_clip_symmetry():
    pass
    db = connect_db('nebula_dev')
    all_movies = get_all_movies(db)
    clip_bench = NebulaVideoEvaluation()
    thresholds = [7.0]

    # query = 'FOR doc IN SemanticStoryNodes_1 return doc'
    # cursor = db.aql.execute(query, ttl=3600)

    # Load sentences
    sentences = MilvusAPI('milvus', 'descriptions', 'nebula_dev', 640)
    # sentences = MilvusAPI('milvus', 'scene_graph', 'nebula_dev', 640)

    for k in range(len(all_movies)):
        movie_name = '/movies/' + all_movies[k]['movie']['file_name'] + '.avi'
        start_time = float(all_movies[k]['movie']['split'])
        end_time = float(all_movies[k]['movie']['splits_total'])
        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds,
                                                                           start_time=start_time, end_time=end_time)

        for cnt in range(embedding_list[0].shape[0]):
            norm_emb = embedding_list[0][cnt, :] / np.linalg.norm(embedding_list[0][cnt, :])
            search_labels = sentences.search_vector(10, norm_emb.tolist())

            sent_list = []

            for distance, data in search_labels:
                # if distance > 0.4:
                print("Context: ", distance, " ", data['sentence'])
                sent_list.append(data['sentence'])

                text_token = torch.cat([clip.tokenize(data['sentence'])]).to('cpu')
                # text_token = torch.cat([clip.tokenize(data['sentence'])]).to('cpu')

                with torch.no_grad():
                    text_features = clip_bench.model.encode_text(text_token)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    emb_torch = torch.from_numpy(norm_emb.reshape((640, 1)))
                    text_features = text_features.type('torch.DoubleTensor')
                    val = (text_features @ emb_torch)

                    text_numpy = text_features.cpu().detach().numpy()
                    print('Distance ', np.matmul(text_numpy, norm_emb.reshape((640, 1))))

            # text_inputs = torch.cat([clip.tokenize(c) for c in sents]).to(self.device)
            # with torch.no_grad():
            #     text_features = self.model_test.encode_text(text_inputs)
            #     text_features /= text_features.norm(dim=-1, keepdim=True)
            # print("Frames vectors nmbr: ", len(embeddings))
            # for i, (image_features, meta) in enumerate(zip(embeddings, metadata)):
            #     print("STAGE: ", i)
            #     print(meta)
            #     # Pick the top 5 most similar labels for the image
            #     # image_features /= image_features.norm(dim=-1, keepdim=True)
            #
            #     similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            #     values, indices = similarity[0].topk(5)
            #
            #     # Print the result
            #     print("\nTop predictions:\n")
            #     for value, index in zip(values, indices):
            #         if (100 * value.item() > 5):
            #             print(f"{sents[index]:>16s}: {100 * value.item():.2f}%")



def test_clip_string_quality():
    """
    The function is used to verify Benchmark results
    :return:
    """
    embedding_loader = EmbeddingsLoader('clip2bert', debug=True)
    string_embedding = EmbeddingsLoader('clip4string', debug=True)

    # Load benchmark data
    sim_meta_data = pd.read_pickle("/home/migakol/data/sim_data.pkl")
    sim_matrix = np.load("/home/migakol/data/sim_benchmark.npy")

    # Load database
    # db = connect_db('nebula_dev')
    # index = MilvusAPI('milvus', 'bert_embeddings', 'nebula_dev', 768)

    similar = embedding_loader.get_similar_movies('Movies/17817709', 10)
    # string_embedding.get_similar_movies('Movies/17808934', 10)
    # embedding_loader.get_bert_id_from_db('Movies/17808934')
    pass


def compare_two_movies(id1, id2):
    db = connect_db('nebula_dev')
    all_movies = get_all_movies(db)

    filename1 = ''
    filename2 = ''
    num1 = 0
    num2 = 0
    for m in all_movies:
        if all_movies[m]['movie']['_id'] == id1:
            filename1 = all_movies[m]['movie']['file_name']
            num1 = m
        if all_movies[m]['movie']['_id'] == id2:
            filename2 = all_movies[m]['movie']['file_name']
            num2 = m

    clip_bench = NebulaVideoEvaluation()

    # thresholds = [7.0]
    # embedding_list1, boundaries1 = clip_bench.create_clip_representation('/movies/' + filename1 + '.avi',
    #                                                                      thresholds=thresholds)
    # embedding_list2, boundaries2 = clip_bench.create_clip_representation('/movies/' + filename2 + '.avi',
    #                                                                      thresholds=thresholds)

    data_folder = '/home/migakol/data'
    file_handle = open(os.path.join(data_folder, 'all_embeddings.pickle'), 'rb')
    list_of_all_embeddings = pickle.load(file_handle)

    sim = clip_bench.find_similarity(list_of_all_embeddings[num1], list_of_all_embeddings[num2])

    pass

def compute_clip_string_similarity():
    embedding_folder = '/home/migakol/data/embedding_folder/'
    # movie_names = '/home/migakol/data/movie_names.pickle'
    num_movies = 3669
    # file_handle = file_handle = open(movie_names, 'rb')
    # name_arr = pickle.load(file_handle)
    #
    # db = connect_db('nebula_datadriven')
    # query = 'FOR doc IN Movies RETURN doc'
    # cursor = db.aql.execute(query, ttl=3600)
    #
    # # Create a list of movie names
    # id_arr = []
    # for k, data in enumerate(cursor):
    #     id_arr.append(data['_id'])
    #     # name_arr.append(data['movie_id'])

    # We assume that name_arr and id_arr have the same order
    meta_data = '/movies/data/clip4string_metadata.pickle'

    # file_handle = open(meta_data, 'rb')
    # pickle.dump([id_arr, name_arr], file_handle)

    file_handle = open(meta_data, 'rb')
    id_arr, name_arr = pickle.load(file_handle)

    similarity_matrix = np.zeros((num_movies, num_movies))
    clip_bench = NebulaVideoEvaluation()
    for k in range(num_movies - 1):
        print('Running movie ', k)
        emb_name = os.path.join(embedding_folder, f'{k:05}.pickle')
        file_handle = open(emb_name, 'rb')
        embedding_list1 = pickle.load(file_handle)

        if len(embedding_list1) == 0:
            print('Movie is bad ', k)
            continue
        if len(embedding_list1[0]) == 0:
            continue
        for m in range(k + 1, num_movies):
            emb_name = os.path.join(embedding_folder, f'{m:05}.pickle')
            file_handle = open(emb_name, 'rb')
            embedding_list2 = pickle.load(file_handle)

            if len(embedding_list2) == 0 or len(embedding_list2[0]) == 0:
                print('Second Movie is bad ', m)
                continue

            sim = clip_bench.find_similarity([embedding_list1[1]], [embedding_list2[1]])
            similarity_matrix[k, m] = sim
            similarity_matrix[m, k] = sim

    f = open('/movies/data/sim_benchmark.npy', 'wb')
    np.save(f, similarity_matrix)
    pass


if __name__ == '__main__':
    print('Started')

    compute_clip_string_similarity()

    # test_clip_string_quality()
    # # test_clip_symmetry()
    #
    # id1 = 'Movies/17808934'
    # id2 = 'Movies/92349441'
    # id3 = 'Movies/92349435'
    #
    # [92349432, 92349432, 92349432, 92349435, 92349435]
    # 92349435, 92349435, 92349438, 92349441, 92349441, 92349441, 92349441, 92349441, 92349441, 92349441
    # # Best bert option
    # # id2 = 'Movies/10969510'
    # compare_two_movies(id1, id2)


    print('Done')
