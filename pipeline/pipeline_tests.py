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

from nebula_api.milvus_api import connect_db
from nebula_api.milvus_api import MilvusAPI


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

        light_house = get_lighthouse_for_movie(movie_name)

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
            concepts, attributes, persons, triplets, verbs = lh_gen.decompose_lighthouse(events=events, actions=actions,
                                                                                  places=places)
            # Save the data to save time
            save_name = f'/home/migakol/data/amr_tests/' + movie_name.split('/')[-1][0:-4] + f'_emb_{k:03d}.pkl'
            f = open(save_name, 'wb')
            pickle.dump([concepts, attributes, persons, triplets, verbs], f)


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
    cluster_words(glove_model, new_concepts)

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
            emb = emb / np.linalg.norm(emb)

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
            concepts, attributes, persons, triplets, verbs = pickle.load(f)
            concepts, attributes, persons, triplets = format_lighthouses(glove_model, concepts, attributes, persons,
                                                                         triplets)
            best_sent, best_res = lh_gen.generate_from_concepts(concepts, attributes, persons, triplets, verbs,
                                                                places, emb)

            save_name = f'/home/migakol/data/amr_tests/res' + movie_name.split('/')[-1][0:-4] + f'_emb_{k:03d}.pkl'
            f = open(save_name, 'wb')
            pickle.dump([best_sent, best_res], f)

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
    generate_sentences_from_lighthouse()

