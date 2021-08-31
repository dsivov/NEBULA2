import pandas as pd
import os
from arango import ArangoClient
import pickle
from benchmark.connection import connect_db

from benchmark.nlp_benchmark import NebulaStoryEvaluation

def create_annotation_data_frame(annotation_file, video_links, jpg_link) -> pd.DataFrame:
    """
    :param annotation_file: - path to the CSV
    :return: data frame with the following columns
    Name
    Start time
    End time
    Video Link
    Collection of frame links
    """

    # Open the files one after another, annotation is the main one
    annotation_file_handler = open(annotation_file, 'r')


    count = 0
    while True:
        count += 1
        # Get next line from file
        annotation_line = annotation_file_handler.readline()
        # parse it
        print(annotation_line)


    video_file_handler = open(video_links, 'r')
    jpg_file_handler = open(jpg_link, 'r')


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
    return ret

def create_lsmdc_dataset(annotation_file):
    """
    Given the annotation file, write the data into the arango DB
    collectio: lsmdc_clips, fields: id, name, start [msec], end [msec], text
    :param annotation_file:
    :return:
    """
    pass

    annotation_file_handler = open(annotation_file, 'r')

    db = connect_db('nebula_datadriven')
    if db.has_collection('lsmdc_clips'):
        db.delete_collection('lsmdc_clips')
    lsmdc_collection = db.create_collection('lsmdc_clips')
    lsmdc_collection.add_hash_index(fields=['_key'])

    id_count = 0
    # Prevent overflow of the id count
    while id_count < 1000000:
        ann_line = annotation_file_handler.readline()

        time_str = ann_line[ann_line.find('\t') - 25:ann_line.find('\t')]
        start_time = time_str[:time_str.find('-')]
        end_time = time_str[time_str.find('-') + 1:]
        start_sec = time_to_sec(start_time)
        end_sec = time_to_sec(end_time)

        # lsmdc_collection.insert({'_key': id_count, 'name': ann_line[0:ann_line.find('\t') - 26], 'start': start_sec, 'end': end_sec, 'text':ann_line[ann_line.find('\t')+1:]})

        lsmdc_collection.insert(
            {'_key': str(id_count), 'name': ann_line[0:ann_line.find('\t') - 26], 'start': str(start_sec),
             'end': str(end_sec), 'text': ann_line[ann_line.find('\t')+1:].rstrip()})

        id_count += 1


def create_text_benchmark(annotation_file, video_links, jpg_link, save_benchmark):
    """
    Create text benchmark based on manual annotatons
    :param annotation_file: path to the CSV file with scene annotations
    :param video_links: path to file with video download links
    :param jpg_link: path to file with image download links
    :param save_benchmark: save data here
    :return:
    """
    # annotations_df = pd.read_csv(annotation_file)
    annotation_file_handler = open(annotation_file, 'r')
    video_file_handler = open(annotation_file, 'r')
    jpg_file_handler = open(annotation_file, 'r')
    count = 0

    while True:
        count += 1

        # Get next line from file
        ann_line = annotation_file_handler.readline()

        # if line is empty
        # end of file is reached
        # if not line:
        #     break
    # Parse the data
    pass

def time_to_sec(time_str):
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


def lsmdc_stats():
    data_folder = '/dataset/lsmdc'
    annotation_file = 'annotations-someone.csv'
    annotation_file_handler = open(os.path.join(data_folder, annotation_file), 'r')

    db = connect_db('nebula_dev')


    count = 0
    diff_hist = []
    while count < 1000000:
        count += 1

        # Get next line from file
        ann_line = annotation_file_handler.readline()

        # if line is empty
        # end of file is reached
        if not ann_line:
            break
        time_str = ann_line[ann_line.find('\t') - 25:ann_line.find('\t')]
        start_time = time_str[:time_str.find('-')]
        end_time = time_str[time_str.find('-') + 1:]
        start_sec = time_to_sec(start_time)
        end_sec = time_to_sec(end_time)

        pass

    print(1)


def debug():
    """
    The function is used for debugging purposes
    :return:
    """
    db = connect_db('nebula_datadriven')
    if db.has_collection('lsmdc_clips'):
        print('Has collection')
    lsmdc_collection = db.collection('lsmdc_clips')
    # cursor = db.aql.execute('FOR doc IN lsmdc_clips FILTER doc._key == 12096264 RETURN doc')
    cursor = db.aql.execute('FOR doc IN lsmdc_clips RETURN doc', ttl=3600)

    lsmdc_movies = [document for document in cursor]
    # lsmdc_collection.g
    pass


def create_siimilarity_matrix():

    db = connect_db('nebula_dev')
    # cursor = db.aql.execute('FOR doc IN lsmdc_clips FILTER doc._key == 12096264 RETURN doc')
    cursor = db.aql.execute('FOR doc IN lsmdc_clips RETURN doc')
    lsmdc_stories = [document['text'] for document in cursor]

    story_eval = NebulaStoryEvaluation()
    sim_matrices, chosen_stories = story_eval.build_similarity_matrices(lsmdc_stories)
    file_handle = open('/home/migakol/data/textbench.pkl', 'wb')
    pickle.dump(sim_matrices, file_handle)
    file_handle = open('/home/migakol/data/chosen_stories.pkl', 'wb')
    pickle.dump(chosen_stories, file_handle)


def evaluate_stories():
    file_handle = open('/home/migakol/data/textbench.pkl', 'rb')
    sim_matrices = pickle.load(file_handle)

    file_handle = open('/home/migakol/data/chosen_stories.pkl', 'rb')
    chosen_stories = pickle.load(file_handle)

    pass

if __name__ == '__main__':
    print('Start textual comparison')

    data_folder = '/dataset/lsmdc'
    annotation_file = 'annotations-someone.csv'
    video_links = 'downloadLinksAvi.txt'
    jpg_links = 'downloadLinksJpg.txt'
    save_benchmark = 'text_benchmark.npy'

    # lsmdc_stats()
    debug()
    create_lsmdc_dataset(os.path.join(data_folder, annotation_file))
    # evaluate_stories()
    # create_siimilarity_matrix()

    # create_text_benchmark(os.path.join(data_folder, annotation_file), save_benchmark)