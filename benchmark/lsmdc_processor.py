import pandas as pd
import os
from arango import ArangoClient
import pickle
from nebula_api.milvus_api import connect_db
from benchmark.clip_benchmark import NebulaVideoEvaluation
from benchmark.nlp_benchmark import NebulaStoryEvaluation
from nebula_api.milvus_api import MilvusAPI
from benchmark.graph_encoder import GraphEncoder
import numpy as np
import csv


def time_to_msec(time_str):
    dot1 = time_str.find('.')
    hours = float(time_str[:dot1])

    time_str = time_str[dot1 + 1:]
    dot1 = time_str.find('.')
    minutes = float(time_str[:dot1])

    time_str = time_str[dot1 + 1:]
    dot1 = time_str.find('.')
    secs = float(time_str[:dot1])

    time_str = time_str[dot1 + 1:]
    mil_sec = float(time_str)

    ret = hours * 60 * 60 * 1000 + minutes * 60 * 1000 + secs * 1000 + mil_sec
    ret = ret / 1000
    return ret


class LSMDCProcessor:
    def __init__(self, annotation_file='annotations-someone.csv'):
        pass
        self.data_folder = '/dataset/lsmdc'
        self.annotation_file = annotation_file
        self.video_links = 'downloadLinksAvi.txt'
        self.jpg_links = 'downloadLinksJpg.txt'
        self.save_benchmark = 'text_benchmark.npy'
        self.db = connect_db('nebula_datadriven')


    def get_all_movies(self):
        query = 'FOR doc IN lsmdc_clips RETURN doc'
        cursor = self.db.aql.execute(query, ttl=3600)

        movies_list = []
        for cnt, movie_data in enumerate(cursor):
            movies_list.append(movie_data)

        return movies_list

    def process_annotation_file(self, annotation_file=''):
        """
        The function populates the database of LSMDC according to the annotation file
        :param annotation_file:
        :return:
        """
        if annotation_file == '':
            annotation_file_handler = open(os.path.join(self.data_folder, self.annotation_file), 'r')
        else:
            annotation_file_handler = open(annotation_file, 'r')

        if self.db.has_collection('lsmdc_clips'):
            self.db.delete_collection('lsmdc_clips')
        lsmdc_collection = self.db.create_collection('lsmdc_clips')
        lsmdc_collection.add_hash_index(fields=['_key'])

        id_count = 0
        # Prevent overflow of the id count
        while id_count < 1000000:
            ann_line = annotation_file_handler.readline()
            if ann_line == '':
                break

            movie_path = ann_line[0:ann_line.find('\t') - 26] + '/' + ann_line[0:ann_line.find('\t')] + '.avi'
            movie_id = ann_line[0:4]
            movie_name = ann_line[5:ann_line.find('\t') - 26]
            time_str = ann_line[ann_line.find('\t') - 25:ann_line.find('\t')]
            start_time = time_str[:time_str.find('-')]
            end_time = time_str[time_str.find('-') + 1:]
            start_sec = time_to_msec(start_time)
            end_sec = time_to_msec(end_time)

            lsmdc_collection.insert(
                {'_key': str(id_count), 'movie_id': movie_id, 'movie_name': movie_name,
                 'name': ann_line[0:ann_line.find('\t') - 26], 'start': str(start_sec),
                 'end': str(end_sec), 'text': ann_line[ann_line.find('\t') + 1:].rstrip(),
                 'path': movie_path
                 })

            id_count += 1


def show_lsmdc_pickle_results():
    outfolder = '/home/migakol/data/lsmdc_test'
    result_file = os.path.join(outfolder, 'result_list1.pickle')
    out_file = os.path.join(outfolder, 'result_csv.csv')

    with open(result_file, 'rb') as f:
        results_list = pickle.load(f)
        with open(out_file, 'w', newline='') as csvfile:
            fieldnames = ['Score', 'Original', 'Chosen']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({'Score': 'Score', 'Original': 'Original', 'Chosen': 'Chosen'})
            for k in range(len(results_list[0])):
                writer.writerow({'Score': results_list[0][k],
                                'Original': results_list[1][k][0:results_list[1][k].find('_MK_')],
                                'Chosen': results_list[1][k][results_list[1][k].find('_MK_') + 4:]})

    pass
    print('Done')


def test_lsmdc_sentences_with_clip():
    """
    The idea is to take a LSMDC movies, convert it into a (number of) CLIP embedding
    and then check the similarity between all the sentences in LSMDC dataset to the
    embedding.
    As a metric, we can use the location of the true ID
    :return:
    """
    pass

    lsmdc_processor = LSMDCProcessor()
    all_movies = lsmdc_processor.get_all_movies()

    video_eval = NebulaVideoEvaluation()
    main_folder = '/dataset/lsmdc/avi/'
    outfolder = '/home/migakol/data/lsmdc_test'
    result_file = os.path.join(outfolder, 'result_list.pickle')

    text_emb_matrix = np.zeros((len(all_movies), 640))
    for k, movie in enumerate(all_movies):
        if len(movie['text']) > 77:
            emb = video_eval.encode_text(movie['text'][0:77])
        else:
            emb = video_eval.encode_text(movie['text'])
        text_emb_matrix[k, :] = emb / np.linalg.norm(emb)

    results_list = []
    sentence_pair = []
    for k, movie in enumerate(all_movies):
        if k < 2:
            continue
        movie_path = main_folder + movie['path']
        embedding_list, boundaries = video_eval.create_clip_representation(movie_path, [0.8])

        embedding_list = embedding_list[0]
        best_pos = len(all_movies)
        for ek in range(embedding_list.shape[0]):
            emb = embedding_list[ek]
            emb = emb / np.linalg.norm(emb)
            res = np.matmul(text_emb_matrix, emb.T)
            res = np.reshape(res, len(res))
            pos = len(res) - np.where(np.argsort(res) == k)[0][0] - 1
            if pos < best_pos:
                best_pos = pos

        results_list.append(best_pos)
        sentence_pair.append(movie['text'] + '_MK_' + all_movies[best_pos]['text'])
        with open(result_file, 'wb') as f:
            pickle.dump([results_list, sentence_pair], f)


# Similarity between CLIP sentences from LSMDC frames and corresponding videos
def clip_based_lsmdc_similarity():
    """
    Go over all lsmdc videos
    :return:
    """
    lsmdc_processor = LSMDCProcessor()
    all_movies = lsmdc_processor.get_all_movies()
    scene_graph = MilvusAPI('milvus', 'scene_graph_visual_genome', 'nebula_dev', 640)
    pegasus_stories = MilvusAPI('milvus', 'pegasus', 'nebula_dev', 640)
    graph_encoder = GraphEncoder()

    video_eval = NebulaVideoEvaluation()
    main_folder = '/dataset/lsmdc/avi/'
    outfolder = '/home/migakol/data/lsmdc_test'
    for movie in all_movies:
        movie_path = main_folder + movie['path']
        embedding_list, boundaries = video_eval.create_clip_representation(movie_path, [0.8])

        # Go over all embeddings in case several are returned
        paragraph_scene = []
        paragraph_pegasus = []
        for emb in embedding_list:
            search_scene_graph = scene_graph.search_vector(1, emb.tolist()[0])
            for distance, data in search_scene_graph:
                paragraph_scene.append(data['sentence'])

            search_scene_graph = pegasus_stories.search_vector(1, emb.tolist()[0])
            for distance, data in search_scene_graph:
                paragraph_pegasus.append(data['sentence'])

        text1 = graph_encoder.encode_with_transformers(paragraph_scene)
        text2 = graph_encoder.encode_with_transformers(paragraph_pegasus)
        text_original = graph_encoder.encode_with_transformers([movie['text']])

        triplet_data = movie['path'][movie['path'].find('/') + 1:] + '.pickle'
        file_handler = open(os.path.join(outfolder, triplet_data), 'wb')
        pickle.dump([text1, text2, text_original, paragraph_scene, paragraph_pegasus, movie['text']], file_handler)



def test_clip_based_lsmdc_similarity():
    data_folder = '/home/migakol/data/lsmdc_test'

    # get the list of files
    files = [f for f in os.listdir(data_folder)]

    # Go over all files
    v1_list = []
    v2_list = []

    for f in files:
        if f[-6:] != 'pickle':
            continue
        file_handler = open(os.path.join(data_folder, f), 'rb')
        enc1, enc2, enc_original = pickle.load(file_handler)
        enc1 = enc1 / np.linalg.norm(enc1)
        enc2 = enc2 / np.linalg.norm(enc2)
        enc_original = enc_original / np.linalg.norm(enc_original)

        v1 = np.sum((enc1.numpy() * enc_original.numpy()))
        v2 = np.sum((enc2.numpy() * enc_original.numpy()))
        v1_list.append(v1)
        v2_list.append(v2)

    pass
    print(np.mean(np.array(v1_list)))
    print(np.mean(np.array(v2_list)))
    print(1)


def create_similarity_analysis():
    data_folder = '/home/migakol/data/lsmdc_test'

    # get the list of files
    files = [f for f in os.listdir(data_folder)]

    # Go over all files
    v1_list = []
    v2_list = []
    v3_list = []

    df = pd.DataFrame(columns=['OrigText', 'Pegasus', 'Triplets', 'Pegasus_Sim', 'Triplet_sim',
                               'TripletToPegasus'])

    k = 0
    for f in files:
        if f[-6:] != 'pickle':
            continue
        file_handler = open(os.path.join(data_folder, f), 'rb')
        enc1, enc2, enc_original, text1, text2, text_orig = pickle.load(file_handler)

        enc1 = enc1 / np.linalg.norm(enc1)
        enc2 = enc2 / np.linalg.norm(enc2)
        enc_original = enc_original / np.linalg.norm(enc_original)

        v1 = np.sum((enc1.numpy() * enc_original.numpy()))
        v2 = np.sum((enc2.numpy() * enc_original.numpy()))
        v3 = np.sum((enc1.numpy() * enc2.numpy()))

        v1_list.append(v1)
        v2_list.append(v2)
        v3_list.append(v3)

        df = df.append({'OrigText': text_orig, 'Pegasus': text2, 'Triplets': text1,
                   'Pegasus_Sim': v2, 'Triplet_sim': v1, 'TripletToPegasus': v3},
                  ignore_index=True)
        k = k + 1
        if k % 100 ==  0:
            print(k)

    df.to_csv(os.path.join(data_folder, 'result.csv'))
    print(v1)
    print(v2)
    print(v3)


def temp_debug():
    outfolder = '/home/migakol/data/lsmdc_test'
    result_file = os.path.join(outfolder, 'result_list.pickle')

    data = pickle.load(open(result_file, 'rb'))

    pass

if __name__ == '__main__':
    print('Start textual comparison')
    # lsmdc_processor = LSMDCProcessor()
    # lsmdc_processor.process_annotation_file()
    # create_similarity_analysis()
    # clip_based_lsmdc_similarity()
    # test_clip_based_lsmdc_similarity()
    # temp_debug()
    show_lsmdc_pickle_results()
    # test_lsmdc_sentences_with_clip()
