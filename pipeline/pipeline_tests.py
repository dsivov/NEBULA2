from benchmark.lsmdc_processor import LSMDCProcessor
from benchmark.clip_benchmark import NebulaVideoEvaluation
import numpy as np
from pipeline.pipeline_master import Pipeline
import experts.tracker.autotracker as at
import os
import shutil
import pickle

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


def run_clip():
    lsmdc_processor = LSMDCProcessor()
    all_movies = lsmdc_processor.get_all_movies()
    small_dataset = LSMDCSmallDataset(lsmdc_processor.db)
    all_ids = small_dataset.get_all_ids()
    clip_bench = NebulaVideoEvaluation()
    base_folder = '/dataset/lsmdc/avi/'
    thresholds = [0.7]
    result_folder = '/home/migakol/data/small_lsmdc_test/clip_results'

    for id in all_ids:
        movie = all_movies[int(id['movie_id'])]
        movie_name = base_folder + movie['path']
        embedding_list, boundaries = clip_bench.create_clip_representation(movie_name, thresholds=thresholds)
        save_name = movie['path'][movie['path'].find('/') + 1:-4] + '_clip.pickle'
        with open(os.path.join(result_folder, save_name), 'wb') as handle:
            pickle.dump([embedding_list, boundaries], handle)


if __name__ == '__main__':
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
    # run_clip()
    create_triplets_from_clip()
