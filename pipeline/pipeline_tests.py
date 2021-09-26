from benchmark.lsmdc_processor import LSMDCProcessor
import numpy as np
from pipeline.pipeline_master import Pipeline
import experts.tracker.autotracker as at

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

    for id in all_ids:
        movie = all_movies[id]
        # Run Tracker on this movie
        tracking_data = at.tracking_utils.MultiTracker.track_video_objects(movie, model, **experiment_config)
        pass


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


if __name__ == '__main__':
    print('Start examples')
    # Part 1 - choose 50 videos
    # create_n_random_lsmdc_dataset(50)
    # Part 2 - run detector and tracker on them
    run_tracker()