from benchmark.lsmdc_processor import LSMDCProcessor
from benchmark.clip_benchmark import NebulaVideoEvaluation
from benchmark.location_list import LocationList
from nlp_tools.light_house_generator import LightHouseGenerator
import numpy as np
from pipeline.pipeline_master import Pipeline
import experts.tracker.autotracker as at
import os
import shutil
import pickle
from gensim.models import KeyedVectors
from pipeline.clip_cap import ClipCap
import csv
import pandas as pd

from nebula_api.milvus_api import connect_db
from nebula_api.milvus_api import MilvusAPI

from nebula_api.atomic2020.comet_enrichment_api import Comet

import time

# from nebula_api.vlmapi import VLM_API
from nebula_api.nebula_enrichment_api import NRE_API

class LSMDCSmallDataset:
    def __init__(self, db):
        self.db = db
        self.small_collection = 'lsmdc_small_dataset'

    def fill_dataset(self, movie_ids: list):
        """
        :param db: StandardDatabase as return by connect_db
        :param movie_ids:
        :return:
        """
        if self.db.has_collection(self.small_collection):
            self.db.delete_collection(self.small_collection)
        lsmdc_collection = self.db.create_collection(self.small_collection)

        for id in movie_ids:
            lsmdc_collection.insert({'movie_id': str(id)})

    def get_all_ids(self):
        query = 'FOR doc IN ' + self.small_collection + ' RETURN doc'
        cursor = self.db.aql.execute(query, ttl=3600)

        id_list = []
        for cnt, movie_data in enumerate(cursor):
            id_list.append(movie_data)

        return id_list



def run_tracker():
    lsmdc_processor = LSMDCProcessor()
    all_movies = lsmdc_processor.get_all_movies()
    small_dataset = LSMDCSmallDataset(lsmdc_processor.db)
    all_ids = small_dataset.get_all_ids()

    pipeline = Pipeline()

    # configure an experiment
    experiment_config = dict(
        detect_every=10,  # use detection model every `detect_every` frames (starting from 0)
        merge_iou_threshold=0.5,  # required IOU score
        tracker_type=at.tracking_utils.TRACKER_TYPE_KCF,  # trecker algorithm (KCF or CSRT)
        refresh_on_detect=False  # if True, any object that isn't found by the model is removed
    )
    model = at.detection_utils.VideoPredictor()
    output_folder = '/home/migakol/data/small_lsmdc_test/tracker_results/'
    base_folder = '/dataset/lsmdc/avi/'
    for id in all_ids:
        movie = all_movies[int(id['movie_id'])]
        # Run Tracker on this movie
        tracking_data = at.tracking_utils.MultiTracker.track_video_objects(base_folder + movie['path'], model,
                                                                           **experiment_config)
        save_name = movie['path'][movie['path'].find('/')+1:-4] + '.pickle'
        with open(os.path.join(output_folder, save_name), 'wb') as handle:
            pickle.dump(tracking_data, handle)
    print('Done saving')


def create_n_random_lsmdc_dataset(N):
    """
    The function choose randomly N movies from LSMDC dataset and create a new dataset
    lsmdc_clips inn nebula_datadriven hold all the innformation about LSMDC movies
    :param N:
    :return:
    """
    lsmdc_processor = LSMDCProcessor()
    all_movies = lsmdc_processor.get_all_movies()
    # random_movies
    random_movies = np.random.permutation(len(all_movies))[0:50]
    # Put them into dataset
    small_dataset = LSMDCSmallDataset(lsmdc_processor.db)
    small_dataset.fill_dataset(random_movies)


def copy_files():
    lsmdc_processor = LSMDCProcessor()
    all_movies = lsmdc_processor.get_all_movies()
    small_dataset = LSMDCSmallDataset(lsmdc_processor.db)
    all_ids = small_dataset.get_all_ids()

    out_folder = '/home/migakol/data/small_lsmdc_test/movies/'
    base_folder = '/dataset/lsmdc/avi/'

    for id in all_ids:
        movie = all_movies[int(id['movie_id'])]
        shutil.copy(base_folder + movie['path'], out_folder)

    print('Done copying')


def run_step():
    lsmdc_processor = LSMDCProcessor()
    all_movies = lsmdc_processor.get_all_movies()
    small_dataset = LSMDCSmallDataset(lsmdc_processor.db)
    all_ids = small_dataset.get_all_ids()

    pipeline = Pipeline()

    # Go over all movies
    movie_folder = '/home/ec2-user/data/movies'
    input_folder = '/home/ec2-user/data/'
    frames_folder = input_folder + 'frames/'
    step_folder = '/home/ec2-user/deployment/STEP'
    res_folder = '/home/ec2-user/data/step_results'
    for id in all_ids:
        movie = all_movies[int(id['movie_id'])]
        movie_filename = movie['path'][movie['path'].find('/') + 1:]
        # The first part is to divide the video into frames and to put it into /frames folder of the input directory
        # Remove all files from the frames folder
        for f in os.listdir(frames_folder):
            if os.path.isfile(os.path.join(frames_folder, f)):
                os.remove(os.path.join(frames_folder, f))
        pipeline.divide_movie_into_frames(os.path.join(movie_folder, movie_filename), frames_folder)

        cmd_line = '/home/ec2-user/miniconda3/envs/michael/bin/python ' + \
                   os.path.join(step_folder, 'demo.py') + ' --input_folder ' + input_folder
        os.system(cmd_line)

        df = pipeline.step_results_postprocessing(os.path.join(input_folder, 'results/results.txt'),
                                              step_folder + '/external/ActivityNet/Evaluation/ava/ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt')
        df.to_pickle(os.path.join(res_folder, movie_filename[:-4] + '.pickle'))


def create_triplets_from_clip():

    # pegasus_stories = MilvusAPI('milvus', 'pegasus', 'nebula_dev', 640)
    scene_graph = MilvusAPI('milvus', 'scene_graph_visual_genome', 'nebula_dev', 640)
    result_folder = '/home/migakol/data/small_lsmdc_test/clip_results'

    # go over all clip embeddings in the folder
    for f in os.listdir(result_folder):
        # Check if it's a file
        if os.path.isfile(os.path.join(result_folder, f)):
            # check if "clip" appears
            if 'clip' in f:
                with open(os.path.join(result_folder, f), 'rb') as handle:
                    emb, _ = pickle.load(handle)
                    paragraph_pegasus = []
                    search_scene_graph = scene_graph.search_vector(1, emb[0].tolist()[0])
                    paragraph_pegasus.append(search_scene_graph[0][1]['sentence'])
                    print(f)
                    print(search_scene_graph[0][1]['sentence'])



    # for emb in embedding_list:
    #     search_scene_graph = scene_graph.search_vector(1, emb.tolist()[0])
    #     for distance, data in search_scene_graph:
    #         paragraph_scene.append(data['sentence'])
    #
    #     search_scene_graph = pegasus_stories.search_vector(1, emb.tolist()[0])
    #     for distance, data in search_scene_graph:
    #         paragraph_pegasus.append(data['sentence'])


def get_text_img_score(text, clip_bench, img_emb):
    text_emb = clip_bench.encode_text(text)
    text_emb = text_emb / np.linalg.norm(text_emb)
    return np.sum((text_emb * img_emb))

def get_text_similarity(text1, text2, clip_bench):
    text_emb1 = clip_bench.encode_text(text1)
    text_emb1 = text_emb1 / np.linalg.norm(text_emb1)
    text_emb2 = clip_bench.encode_text(text2)
    text_emb2 = text_emb2 / np.linalg.norm(text_emb2)
    return np.sum((text_emb1 * text_emb2))

def test_clip_single_image():
    import cv2 as cv
    from PIL import Image
    clip_bench = NebulaVideoEvaluation()
    img_name = '/home/migakol/data/img1.jpg'

    frame = cv.imread(img_name)
    img = clip_bench.preprocess(Image.fromarray(frame)).unsqueeze(0).to('cpu')
    img_emb = clip_bench.model.encode_image(img).detach().numpy()
    img_emb = img_emb / np.linalg.norm(img_emb)

    text = ''
    text_emb = clip_bench.encode_text(text)
    text_emb = text_emb / np.linalg.norm(text_emb)




    print(1)

def relative_distances(all_embeddings):
    all_embeddings_vec = np.array(all_embeddings)
    cos_dist = np.zeros((all_embeddings_vec.shape[0], all_embeddings_vec.shape[0]))
    reg_dist = np.zeros((all_embeddings_vec.shape[0], all_embeddings_vec.shape[0]))
    for x in range(all_embeddings_vec.shape[0] - 1):
        for y in range(x, all_embeddings_vec.shape[0]):
            v1 = all_embeddings_vec[x, :] / np.linalg.norm(all_embeddings_vec[x, :])
            v2 = all_embeddings_vec[y, :] / np.linalg.norm(all_embeddings_vec[y, :])
            cos_dist[y, x] = np.dot(v1, v2)
            cos_dist[x, y] = np.dot(v1, v2)
            reg_dist[y, x] = np.linalg.norm(all_embeddings_vec[x, :] - all_embeddings_vec[y, :])
            reg_dist[x, y] = reg_dist[y, x]

    return cos_dist, reg_dist


def get_small_movies_ids():
    lsmdc_processor = LSMDCProcessor()
    all_movies = lsmdc_processor.get_all_movies()
    small_dataset = LSMDCSmallDataset(lsmdc_processor.db)
    all_ids = small_dataset.get_all_ids()

    return all_movies, all_ids

def run_clip():
    all_movies, all_ids = get_small_movies_ids()

    clip_bench = NebulaVideoEvaluation()
    base_folder = '/dataset/lsmdc/avi/'
    thresholds = [0.8]
    result_folder = '/home/migakol/data/small_lsmdc_test/clip_results'

    # scene_graph = MilvusAPI('milvus', 'scene_graph_visual_genome', 'nebula_dev', 640)
    scene_graph = MilvusAPI('milvus', 'pegasus', 'nebula_dev', 640)

    location_list = LocationList()

    db = connect_db('nebula_development')
    query = 'FOR doc IN nebula_vcomet_lighthouse RETURN doc'
    cursor = db.aql.execute(query, ttl=3600)
    for doc in cursor:
        # print(doc)
        if doc['url_link'].split('/')[-1] == '1031_Quantum_of_Solace_00_52_35_159-00_52_37_144.mp4':
            print(1)
        print(doc)

    paragraph_pegasus = []
    all_embeddings = []
    for id in all_ids:
        movie = all_movies[int(id['movie_id'])]
        movie_name = base_folder + movie['path']
        print(movie['path'])
        # if movie_name != '/dataset/lsmdc/avi/1031_Quantum_of_Solace/1031_Quantum_of_Solace_00.39.09.510-00.39.14.286.avi':
        if movie_name != '/dataset/lsmdc/avi/1031_Quantum_of_Solace/1031_Quantum_of_Solace_00.39.09.510-00.39.14.286.avi':
            continue
        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds, method='single')
        save_name = movie['path'][movie['path'].find('/') + 1:-4] + '_clip.pickle'
        # with open(os.path.join(result_folder, save_name), 'wb') as handle:
        #     pickle.dump([embedding_list, boundaries], handle)

        paragraph_pegasus.append(movie['path'])
        movie_locations = []
        for k in range(embedding_list[0].shape[0]):
            emb = embedding_list[0][k, :]
            all_embeddings.append(emb)
            search_scene_graph = scene_graph.search_vector(20, emb.tolist())
            paragraph_pegasus.append('SECTION ' + str(boundaries[0][k]))
            for x in range(20):
                paragraph_pegasus.append(str(search_scene_graph[x][0]) + '   ' + search_scene_graph[x][1]['sentence'])
            print(search_scene_graph[0][1]['sentence'])

            max_score = 0
            best_ind = 0
            for loc_ind, loc in enumerate(location_list.locations):
                score = get_text_img_score('it is a ' + loc, clip_bench, emb)
                # score = get_text_img_score(search_scene_graph[0][1]['sentence'] + ' in a ' + loc, clip_bench, emb)
                if score > max_score:
                    max_score = score
                    best_ind = loc_ind
            movie_locations.append(location_list.locations[best_ind])


        save_name = movie['path'][movie['path'].find('/') + 1:-4] + '_text_single.pickle'
        handle = open(os.path.join(result_folder, save_name), 'wb')
        pickle.dump(paragraph_pegasus, handle)

        save_name = movie['path'][movie['path'].find('/') + 1:-4] + '_location.pickle'
        with open(os.path.join(result_folder, save_name), 'wb') as handle:
            pickle.dump(movie_locations, handle)

    print(paragraph_pegasus)
    cos_dist, reg_dist = relative_distances(all_embeddings)



    text_results = 'all_text_single.pickle'
    with open(os.path.join(result_folder, text_results), 'wb') as handle:
        pickle.dump(paragraph_pegasus, handle)

    print('Working on median')
    paragraph_pegasus = []
    for id in all_ids:
        movie = all_movies[int(id['movie_id'])]
        movie_name = base_folder + movie['path']
        print(movie['path'])
        if movie_name != '/dataset/lsmdc/avi/1024_Identity_Thief/1024_Identity_Thief_00.01.43.655-00.01.47.807.avi':
            continue
        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds,
                                                                           method='median')
        save_name = movie['path'][movie['path'].find('/') + 1:-4] + '_clip.pickle'
        with open(os.path.join(result_folder, save_name), 'wb') as handle:
            pickle.dump([embedding_list, boundaries], handle)

        paragraph_pegasus.append(movie['path'])
        for k in range(embedding_list[0].shape[0]):
            emb = embedding_list[0][k, :]
            search_scene_graph = scene_graph.search_vector(20, emb.tolist())
            paragraph_pegasus.append('SECTION ' + str(boundaries[0][k]))
            for x in range(20):
                paragraph_pegasus.append(str(search_scene_graph[x][0]) + '   ' + search_scene_graph[x][1]['sentence'])
            print(search_scene_graph[0][1]['sentence'])

        save_name = movie['path'][movie['path'].find('/') + 1:-4] + '_text_median.pickle'
        handle = open(os.path.join(result_folder, save_name), 'wb')
        pickle.dump(paragraph_pegasus, handle)

    print(paragraph_pegasus)

    text_results = 'all_text_median.pickle'
    with open(os.path.join(result_folder, text_results), 'wb') as handle:
        pickle.dump(paragraph_pegasus, handle)


def get_locations_for_lsmdc_movies():
    """
    Go over the small LSMDC subtest and find the optimal location for each one
    :return:
    """
    lsmdc_processor = LSMDCProcessor()
    all_movies = lsmdc_processor.get_all_movies()
    small_dataset = LSMDCSmallDataset(lsmdc_processor.db)
    all_ids = small_dataset.get_all_ids()
    clip_bench = NebulaVideoEvaluation()
    base_folder = '/dataset/lsmdc/avi/'
    thresholds = [0.8]
    result_folder = '/home/migakol/data/small_lsmdc_test/clip_results'
    location_list = LocationList()
    scene_graph = MilvusAPI('milvus', 'scene_graph_visual_genome', 'nebula_dev', 640)

    paragraph_pegasus = []
    for id in all_ids:
        movie = all_movies[int(id['movie_id'])]
        movie_name = base_folder + movie['path']
        print(movie['path'])
        # if movie_name != '/dataset/lsmdc/avi/1010_TITANIC/1010_TITANIC_00.41.32.072-00.41.40.196.avi':
        #     continue
        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds,
                                                                           method='single')

        # Go over all clip embeddings and choose the best location for each one
        movie_locations = []
        for k in range(embedding_list[0].shape[0]):
            emb = embedding_list[0][k, :]
            max_score = 0
            best_ind = 0
            for emb_k, loc_emb in enumerate(location_list.loc_emb):
                score = np.sum((loc_emb * emb))
                if score > max_score:
                    max_score = score
                    best_ind = k
            movie_locations.append(location_list.locations[best_ind])

        save_name = movie['path'][movie['path'].find('/') + 1:-4] + '_location.pickle'
        with open(os.path.join(result_folder, save_name), 'wb') as handle:
            pickle.dump(movie_locations, handle)


def test_locations():
    result_folder = '/home/migakol/data/small_lsmdc_test/clip_results'

    for f in os.listdir(result_folder):
        if 'location' not in f:
            continue
        with open(os.path.join(result_folder, f), 'rb') as handle:
            print(f)
            movie_locations = pickle.load(handle)
            print(movie_locations)
            pass


def test_locations2():
    result_folder = '/home/migakol/data/small_lsmdc_test/clip_results'

    for f in os.listdir(result_folder):
        if 'location' in f:
            continue
        if 'text' not in f:
            continue
        with open(os.path.join(result_folder, f), 'rb') as handle:
            print(f)
            movie_locations = pickle.load(handle)
            print(movie_locations)
            pass


def get_lighthouse_for_movie(movie_name):
    db_nebula_dev = connect_db('nebula_development')
    base_address = '\'http://ec2-3-120-189-231.eu-central-1.compute.amazonaws.com:7000/static/development/'
    # query = 'FOR doc IN nebula_vcomet_lighthouse RETURN doc'

    # Get the based movie name (without location in the movie)
    full_url = base_address + movie_name.split('/')[-1].replace('.', '_')[:-4] + '.mp4\''

    query = 'FOR doc IN nebula_vcomet_lighthouse FILTER doc.url_link == ' + full_url + ' RETURN doc'
    # query = 'FOR doc IN nebula_vcomet_lighthouse RETURN doc'

    cursor = db_nebula_dev.aql.execute(query, ttl=3600)
    doc_list = []
    for doc in cursor:
        doc_list.append(doc)

    # Rearrange the output structure. Here it is
    # doc_list[0]['events'], doc_list[0]['actions'], doc_list[0]['places']
    # doc_list[0]['events'], doc_list[1]['events']

    return doc_list




def save_lighthouse_components():
    all_movies, all_ids = get_small_movies_ids()
    clip_bench = NebulaVideoEvaluation()
    thresholds = [0.8]
    base_folder = '/dataset/lsmdc/avi/'

    # db_nebula_dev = connect_db('nebula_development')
    # query = 'FOR doc IN nebula_vcomet_lighthouse RETURN doc'
    # cursor = db_nebula_dev.aql.execute(query, ttl=3600)
    # doc_list = []
    # for doc in cursor:
    #     doc_list.append(doc)

    lh_gen = LightHouseGenerator()

    # Go over all movies
    for id in all_ids:
        movie = all_movies[int(id['movie_id'])]
        movie_name = base_folder + movie['path']
        print(movie['path'])

        # if movie['path'] != '1015_27_Dresses/1015_27_Dresses_00.38.02.757-00.38.08.213.avi':
        #     continue

        start = time.time()
        light_house = get_lighthouse_for_movie(movie_name)
        end = time.time()
        print('Get lighthouses ', end - start)

        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds,
                                                                           method='single')
        # Go over all embeddings
        for k in range(embedding_list[0].shape[0]):
            emb = embedding_list[0][k, :]

            # Slightly change the lighthouse format
            if k >= len(light_house):
                continue
            events = [ev[1][0] for ev in light_house[k]['events']]
            actions = [ac[1][0] for ac in light_house[k]['actions']]
            places = [pl[1] for pl in light_house[k]['places']]
            # Find the relevant lighthouses in the database
            start = time.time()
            concepts, attributes, persons, triplets, verbs, all_concept, all_dicts = lh_gen.decompose_lighthouse(
                events=events, actions=actions, places=places)
            end = time.time()
            print('Lighthouse decompose ', end - start)
            # Save the data to save time
            save_name = f'/home/migakol/data/amr_tests/' + movie_name.split('/')[-1][0:-4] + f'_emb_{k:03d}.pkl'
            f = open(save_name, 'wb')
            pickle.dump([concepts, attributes, persons, triplets, verbs, all_concept, all_dicts], f)


def cluster_words(glove_model, concepts):
    """
    Given a list of concepts, combine them into groups
    :param glove_model:
    :param concepts:
    :return:
    """
    concepts = list(set(concepts))
    updated_concepts = []
    update_vectors = np.zeros((100, 0))
    for concept in concepts:
        if 'Person' in concept or 'person' in concept:
            continue
        updated_concepts.append(concept)
        w_vector = np.zeros((100, 1))
        w_n = 0
        for w in concept.split(' '):
            if w == '':
                continue
            w_vector = w_vector + glove_model.get_vector(w).reshape((100, 1))
            w_n += 1
        w_vector = w_vector / w_n
        print(np.linalg.norm(w_vector))
        w_vector = w_vector / np.linalg.norm(w_vector)
        update_vectors = np.append(update_vectors, w_vector, axis=1)

    dist_mat = np.matmul(update_vectors.T, update_vectors)

def format_lighthouses(glove_model, concepts, attributes, persons, triplets):
    new_concepts = []
    new_attributes = []
    new_persons = []
    new_triplets = []
    for k in concepts.keys():
        new_concepts = new_concepts + concepts[k]
    for k in attributes.keys():
        new_attributes = new_attributes + attributes[k]
    for k in persons.keys():
        new_persons = new_persons + persons[k]
    for k in triplets.keys():
        new_triplets = new_triplets + triplets[k]

    # Go over all concepts
    # cluster_words(glove_model, new_concepts)

    return list(set(new_concepts)), list(set(new_attributes)), list(set(new_persons)), list(set(new_triplets))



def generate_sentences_from_lighthouse():
    all_movies, all_ids = get_small_movies_ids()
    clip_bench = NebulaVideoEvaluation()
    low_res_clip_bench = NebulaVideoEvaluation('ViT-B/32')
    thresholds = [0.8]
    base_folder = '/dataset/lsmdc/avi/'

    glove_filename = '/home/migakol/data/glove/glove.6B.100d.txt'
    word2vec_output_file = glove_filename + '.word2vec'
    glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    clip_cap = ClipCap()
    # from gensim.scripts.glove2word2vec import glove2word2vec
    # glove2word2vec(glove_filename, word2vec_output_file)

    # db_nebula_dev = connect_db('nebula_development')
    # query = 'FOR doc IN nebula_vcomet_lighthouse RETURN doc'
    # cursor = db_nebula_dev.aql.execute(query, ttl=3600)
    # doc_list = []
    # for doc in cursor:
    #     doc_list.append(doc)

    lh_gen = LightHouseGenerator()

    # Go over all movies
    for id in all_ids:
        movie = all_movies[int(id['movie_id'])]
        movie_name = base_folder + movie['path']
        print(movie['path'])

        if movie['path'] != '1015_27_Dresses/1015_27_Dresses_00.38.02.757-00.38.08.213.avi':
            continue

        light_house = get_lighthouse_for_movie(movie_name)

        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds,
                                                                           method='single')
        # Go over all embeddings
        for k in range(embedding_list[0].shape[0]):
            emb = embedding_list[0][k, :]

            clip_cap_emb = low_res_clip_bench.get_leg_representaion(movie_name, boundaries[0][k], method='single')
            clip_cap_text = clip_cap.generate_text(clip_cap_emb)
            clipcap_emb = clip_bench.encode_text(clip_cap_text)
            clipcap_emb = clipcap_emb / np.linalg.norm(clipcap_emb)
            emb = emb / np.linalg.norm(emb)
            clip_cap_res = np.matmul(clipcap_emb, emb)
            print('Clip cap res ', clip_cap_text, ' ', clip_cap_res)

            events = [ev[1][0] for ev in light_house[k]['events']]
            actions = [ac[1][0] for ac in light_house[k]['actions']]
            places = [pl[1] for pl in light_house[k]['places']]

            # Compute the best lighthouse event
            best_res = 0
            best_sent = ''
            for event in events:
                sent_emb = clip_bench.encode_text(event)
                sent_emb = sent_emb / np.linalg.norm(sent_emb)
                res = np.sum((emb * sent_emb))

                if res > best_res:
                    best_sent = event

            print('Best lighthouse sent ', best_sent)
            print('Best lighthouse res ', best_res)

            # Slightly change the lighthouse format
            if k >= len(light_house):
                continue
            # Load the saved data
            load_name = f'/home/migakol/data/amr_tests/' + movie_name.split('/')[-1][0:-4] + f'_emb_{k:03d}.pkl'
            f = open(load_name, 'rb')
            concepts, attributes, persons, triplets, verbs, all_concept, all_dicts = pickle.load(f)
            concepts, attributes, persons, triplets = format_lighthouses(glove_model, concepts, attributes, persons,
                                                                         triplets)
            best_sent, best_res = lh_gen.generate_from_concepts(concepts, attributes, persons, triplets, verbs,
                                                                places, emb, mode='generate_and_test')

            save_name = f'/home/migakol/data/amr_tests/res' + movie_name.split('/')[-1][0:-4] + f'_emb_{k:03d}.pkl'
            f = open(save_name, 'wb')
            pickle.dump([best_sent, best_res], f)

            pass


def find_movie(dima_movie_name, movie_list):
    for movie in movie_list:
        pass
        if movie['name'] in dima_movie_name:
            sep = movie['path'].split('.')
            dima_sep = dima_movie_name.split('.')[0].split('_')
            if sep[-4] == dima_sep[-3] and sep[-3] == dima_sep[-2] and sep[-2] == dima_sep[-1]:
                return movie['path']


def fill_clip_dataset_with_lsmdc():
    low_res_clip_bench = NebulaVideoEvaluation('ViT-B/32')
    base_folder = '/dataset/lsmdc/avi/'

    comet = Comet("/home/migakol/data/comet/comet-atomic_2020_BART")
    movie_ids = comet.get_playground_movies()

    clipcap_db = comet.db.collection('nebula_clipcap_lighthouse_lsmdc')
    all_movies, all_ids = get_small_movies_ids()

    # query_r = 'FOR doc IN nebula_clipcap_lighthouse_lsmdc RETURN doc'
    # cursor_r = comet.db.aql.execute(query_r, ttl=3600)
    # stages = []
    # for stage in cursor_r:
    #     stages.append(stage)

    clip_cap = ClipCap()

    for id in movie_ids:
        print('Movie ', id)
        stages = comet.get_stages(id)
        for stage in stages:
            boundary = (stage['start'], stage['stop'])
            movie_names = stage['full_path'].split('/')
            if movie_names[0] == '':
                movie_name = stage['full_path'].split('/')[3]
            else:
                movie_name = stage['full_path'].split('/')[2]

            movie_path = find_movie(movie_name, all_movies)
            movie_full_path = base_folder + movie_path
            clip_cap_emb = low_res_clip_bench.get_leg_representaion(movie_full_path, boundary, method='single')
            clip_cap_text = clip_cap.generate_text(clip_cap_emb)

            clipcap_db.insert({'arango_id': id, 'sentence': clip_cap_text, 'scene_element': stage['scene_element']})

            pass
        # benchmark_collection.insert({'index1': k, 'index2': m, 'sim_value': similarity_matrix[k, m]})

        pass


def generate_clipcap_for_test_movies():
    """
    Go over all test movies and generate ClipCap sentence for each one of them
    :return:
    """
    all_movies, all_ids = get_small_movies_ids()
    clip_bench = NebulaVideoEvaluation()
    low_res_clip_bench = NebulaVideoEvaluation('ViT-B/32')
    # low_res_clip_bench = NebulaVideoEvaluation()
    thresholds = [0.8]
    clip_cap = ClipCap()
    base_folder = '/dataset/lsmdc/avi/'
    comet = Comet("/home/migakol/data/comet/comet-atomic_2020_BART")
    if comet.db.has_collection('nebula_clipcap_results'):
        clipcap_db = comet.db.collection('nebula_clipcap_results')
    else:
        clipcap_db = comet.db.create_collection('nebula_clipcap_results')

    # comet.db.delete_collection('nebula_clipcap_results')

    for movie_cnt, movie in enumerate(all_movies):
        # We are interested only in these movies
        if movie['movie_name'] != 'Juno' and movie['movie_name'] != 'Unbreakable' and \
                movie['movie_name'] != 'Bad_Santa' and movie['movie_name'] != 'Super_8' and \
                movie['movie_name'] != 'The_Ugly_Truth' and movie['movie_name'] != 'This_is_40' and \
                movie['movie_name'] != 'Harry_Potter_and_the_prisoner_of_azkaban':
            continue

        if movie['movie_name'] == 'Juno':
            continue

        movie_name = base_folder + movie['path']
        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds,
                                                                       method='single')
        # Go over all embeddings
        for k in range(embedding_list[0].shape[0]):
            emb = embedding_list[0][k, :]

            clip_cap_emb = low_res_clip_bench.get_leg_representaion(movie_name, boundaries[0][k], method='single')
            clip_cap_text = clip_cap.generate_text(clip_cap_emb, use_beam_search=False, num_versions=3)
            # clip_cap_text = clip_cap.generate_text(clip_cap_emb, use_beam_search=False)

            clipcap_db.insert({'lsmdc_id': movie['_key'], 'sentence0': clip_cap_text[0],
                               'sentence1': clip_cap_text[1], 'sentence2': clip_cap_text[2], 'scene_element': k,
                               'path': movie['path']})

        with open('/home/migakol/data/count.txt', 'w') as f:
            f.write(str(movie_cnt) + ' from ' + str(len(all_movies)))


def fill_clipcap():
    all_movies, all_ids = get_small_movies_ids()
    thresholds = [0.8]
    base_folder = '/dataset/lsmdc/avi/'
    clip_bench = NebulaVideoEvaluation()
    clip_cap = ClipCap()
    low_res_clip_bench = NebulaVideoEvaluation('ViT-B/32')

    for id in all_ids:
        movie = all_movies[int(id['movie_id'])]
        movie_name = base_folder + movie['path']
        print(movie['path'])

        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds,
                                                                           method='single')
        # Go over all embeddings
        for k in range(embedding_list[0].shape[0]):
            emb = embedding_list[0][k, :]

            load_name = f'/home/migakol/data/amr_tests/' + movie_name.split('/')[-1][0:-4] + f'_emb_{k:03d}.pkl'
            if not os.path.isfile(load_name):
                continue

            clip_cap_emb = low_res_clip_bench.get_leg_representaion(movie_name, boundaries[0][k], method='single')
            clip_cap_text = clip_cap.generate_text(clip_cap_emb)

            f = open(load_name, 'rb')
            loaded_data = pickle.load(f)
            triplets = loaded_data[3]
            print('Loaded')
            # concepts, attributes, persons, triplets, verbs, all_concept, all_dicts = pickle.load(f)


def get_clipcap_test_movies():
    all_movies, all_ids = get_small_movies_ids()

    data = []
    for movie in all_movies:
        # We are interested only in these movies
        if movie['movie_name'] != 'Juno' and movie['movie_name'] != 'Unbreakable' and \
                movie['movie_name'] != 'Bad_Santa' and movie['movie_name'] != 'Super_8' and \
                movie['movie_name'] != 'The_Ugly_Truth' and movie['movie_name'] != 'This_is_40' and \
                movie['movie_name'] != 'Harry_Potter_and_the_prisoner_of_azkaban':
            continue
        data.append([movie['path'], movie['text']])

    movies_df = pd.DataFrame(data, columns=['path', 'text'])

    return movies_df



def create_csv_file_from_clipcap_results():
    # pass
    comet = Comet("/home/migakol/data/comet/comet-atomic_2020_BART")
    # if comet.db.has_collection('nebula_clipcap_results'):
    #     clipcap_db = comet.db.collection('nebula_clipcap_results')
    # else:
    #     print('Error')
    #     return

    query = 'FOR doc IN nebula_clipcap_results RETURN doc'
    cursor = comet.db.aql.execute(query, ttl=3600)

    movies_list = []
    movie_clipcap_data = []
    for cnt, movie_data in enumerate(cursor):
        movies_list.append(movie_data)
        movie_clipcap_data.append([movie_data['path'], movie_data['sentence'], movie_data['scene_element']])

    full_movies_df = get_clipcap_test_movies()
    movies_df = pd.DataFrame(movie_clipcap_data, columns=['path', 'sentence', 'scene_element'])
    result = movies_df.merge(full_movies_df, on=['path', 'path'])

    outfolder = '/home/migakol/data'
    out_file = os.path.join(outfolder, 'clipcap_result_csv.csv')

    # with open(out_file, 'w', newline='') as csvfile:
    #     fieldnames = ['path', 'sentence', 'scene_element']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #
    #     writer.writerow({'path': 'path', 'sentence': 'sentence', 'scene_element': 'scene_element'})
    #     for movie in movies_list:
    #         writer.writerow({'path': movie['path'],
    #                          'sentence': movie['sentence'],
    #                          'scene_element': movie['scene_element']})



    return movies_list



def test_vcomet():
    clip_bench = NebulaVideoEvaluation(model_name='ViT-L/14')
    # nre = NRE_API()
    video_path = '/dataset/lsmdc/avi/2041_This_is_40/2041_This_is_40_01.29.37.623-01.29.41.335.avi'
    thresholds = [0.8]

    milvus_actions = MilvusAPI(
        'milvus', 'vcomet_vit_embedded_actions', 'nebula_visualcomet', 768)
    milvus_places = MilvusAPI(
        'milvus', 'vcomet_vit_embedded_place', 'nebula_visualcomet', 768)


    base_path = '/dataset/lsmdc/avi/'
    video_path_list = ['1004_Juno/1004_Juno_00.01.39.976-00.01.44.938.avi',
                       '1004_Juno/1004_Juno_00.04.14.990-00.04.16.880.avi',
                       '1004_Juno/1004_Juno_00.04.41.434-00.04.44.257.avi',
                       '1004_Juno/1004_Juno_00.04.51.003-00.05.07.048.avi',
                       '1004_Juno/1004_Juno_00.08.20.770-00.08.29.282.avi',
                       '1004_Juno/1004_Juno_00.08.37.329-00.08.41.913.avi',
                       '1004_Juno/1004_Juno_00.09.40.941-00.09.44.478.avi',
                       '1004_Juno/1004_Juno_00.09.46.160-00.09.49.150.avi',
                       '1004_Juno/1004_Juno_00.09.49.670-00.09.52.513.avi',
                       '1004_Juno/1004_Juno_00.10.20.997-00.10.28.000.avi',
                       '1004_Juno/1004_Juno_00.17.47.985-00.17.51.984.avi',
                       '1004_Juno/1004_Juno_00.17.52.537-00.17.55.547.avi',
                       '1004_Juno/1004_Juno_00.17.55.617-00.18.01.875.avi',
                       '1004_Juno/1004_Juno_00.18.27.074-00.18.30.270.avi',
                       '1004_Juno/1004_Juno_00.18.47.870-00.18.51.230.avi',
                       '1004_Juno/1004_Juno_00.18.53.200-00.19.02.487.avi',
                       '1004_Juno/1004_Juno_00.19.06.160-00.19.12.291.avi',
                       '1004_Juno/1004_Juno_00.57.09.876-00.57.17.607.avi',
                       '1004_Juno/1004_Juno_00.57.28.185-00.57.32.714.avi',
                       '1004_Juno/1004_Juno_00.57.33.660-00.57.34.407.avi',
                       '1004_Juno/1004_Juno_00.57.56.390-00.57.57.890.avi',
                       '1004_Juno/1004_Juno_00.58.08.667-00.58.12.217.avi',
                       '1004_Juno/1004_Juno_00.58.19.120-00.58.22.023.avi',
                       '1004_Juno/1004_Juno_00.58.36.730-00.58.40.397.avi',
                       '1004_Juno/1004_Juno_00.58.40.730-00.58.46.362.avi',
                       '1004_Juno/1004_Juno_00.58.46.362-00.58.49.836.avi',
                       '1004_Juno/1004_Juno_00.59.23.732-00.59.26.152.avi',
                       '1004_Juno/1004_Juno_00.59.26.393-00.59.41.540.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.15.04.415-00.15.05.751.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.15.12.845-00.15.14.997.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.15.45.102-00.15.48.769.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.15.51.900-00.15.59.406.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.15.59.406-00.16.03.780.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.16.17.955-00.16.30.078.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.27.48.061-00.27.48.833.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.31.01.623-00.31.03.873.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.31.14.267-00.31.21.031.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.31.21.053-00.31.24.666.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.32.01.346-00.32.09.016.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.32.37.157-00.32.40.824.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.32.41.157-00.32.43.824.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.32.44.157-00.32.45.824.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.32.46.157-00.32.47.824.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.32.48.157-00.32.50.824.avi',
                       '1012_Unbreakable/1012_Unbreakable_00.32.51.157-00.32.55.855.avi',
                       '1017_Bad_Santa/1017_Bad_Santa_01.13.12.781-01.13.15.464.avi',
                       '1017_Bad_Santa/1017_Bad_Santa_01.13.30.602-01.13.34.269.avi',
                       '1017_Bad_Santa/1017_Bad_Santa_01.13.43.892-01.13.47.327.avi',
                       '1017_Bad_Santa/1017_Bad_Santa_01.13.54.972-01.13.56.385.avi',
                       '1017_Bad_Santa/1017_Bad_Santa_01.14.10.052-01.14.12.195.avi',
                       '1017_Bad_Santa/1017_Bad_Santa_01.14.19.107-01.14.19.774.avi',
                       '1017_Bad_Santa/1017_Bad_Santa_01.14.21.852-01.14.23.355.avi',
                       '1017_Bad_Santa/1017_Bad_Santa_01.14.27.602-01.14.28.785.avi',
                       '1017_Bad_Santa/1017_Bad_Santa_01.14.34.612-01.14.36.425.avi',
                       '1017_Bad_Santa/1017_Bad_Santa_01.14.42.852-01.14.44.519.avi']

    action_list = []
    place_list = []
    LEN = 50
    for video in video_path_list:
        video_path = base_path + video
        embedding_list, boundaries = clip_bench.create_clip_representation(video_path, thresholds=thresholds,
                                                                   method='single')
        for k, emb in enumerate(embedding_list):
            similar_nodes = milvus_actions.search_vector(LEN, emb.tolist()[0])
            print(similar_nodes)
            similar_place = milvus_places.search_vector(LEN, emb.tolist()[0])
            print(similar_place)

            for sim_node in similar_nodes:
                action_list.append([video, k, sim_node[0], sim_node[1]['sentence']])

            for sim_node in similar_place:
                place_list.append([video, k, sim_node[0], sim_node[1]['sentence']])

    action_df = pd.DataFrame(action_list, columns=['video', 'scene_element', 'score', 'text'])
    place_df = pd.DataFrame(place_list, columns=['video', 'scene_element', 'score', 'text'])

    action_df.to_csv(os.path.join('/home/migakol/data', 'actions_res.csv'))
    place_df.to_csv(os.path.join('/home/migakol/data', 'place_res.csv'))

def small_tests():
    video_path = '/dataset/lsmdc/avi/1004_Juno/1004_Juno_00.04.14.990-00.04.16.880.avi'
    thresholds = [0.8]
    clip_bench = NebulaVideoEvaluation(model_name='ViT-L/14')
    embedding_list, boundaries = clip_bench.create_clip_representation(video_path, thresholds=thresholds,
                                                                   method='single')
    low_res_clip_bench = NebulaVideoEvaluation('ViT-B/32')
    clip_cap = ClipCap()
    # from gensim.scripts.glove2word2ve

    for k in range(len(boundaries[0])):
        clip_cap_emb = low_res_clip_bench.get_leg_representaion(video_path, boundaries[0][k], method='single')
        clip_cap_text = clip_cap.generate_text(clip_cap_emb, use_beam_search=True)
        pass

if __name__ == '__main__':

    # text_results = 'all_text_single.pickle'
    # with open(os.path.join(result_folder, text_results), 'wb') as handle:
    #     pickle.dump(paragraph_pegasus, handle)

    print('Start examples')
    # Part 1 - choose 50 videos
    # create_n_random_lsmdc_dataset(50)
    # Part 2 - run detector and tracker on them
    # run_tracker()
    # auxilary function that copies all relevant file to one folder
    # copy_files()
    # Part 3 - run STEP on the files. Note that this part runs on a different computer - GPU
    # run_step()
    # Part 4 - run CLIP on all the data
    # test_clip_single_image()
    # run_clip()
    # get_locations_for_lsmdc_movies()
    # test_locations()
    # create_triplets_from_clip()

    # Lighthouse fusion
    # part 1 - generate concepts, attributes, and persons to speed up the subsequent lighthouse computations
    # It is done, by decomposing the lighthouse into components
    # save_lighthouse_components()
    # part 2 - load the concepts and use them to generate permutations and compare them against CLIP
    # generate_sentences_from_lighthouse()
    # fill_clip_dataset_with_lsmdc()

    # fill_clipcap()

    # Generate CLIP CAP and GRAPH data for all test movies
    generate_clipcap_for_test_movies()
    # create_csv_file_from_clipcap_results()
    # test_vcomet()
    # small_tests()
