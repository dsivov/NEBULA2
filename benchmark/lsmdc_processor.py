import pandas as pd
import os
from arango import ArangoClient
import pickle
from benchmark.connection import connect_db
from benchmark.nlp_benchmark import NebulaStoryEvaluation


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

        db = connect_db('nebula_datadriven')
        if db.has_collection('lsmdc_clips'):
            db.delete_collection('lsmdc_clips')
        lsmdc_collection = db.create_collection('lsmdc_clips')
        lsmdc_collection.add_hash_index(fields=['_key'])

        id_count = 0
        # Prevent overflow of the id count
        while id_count < 1000000:
            ann_line = annotation_file_handler.readline()
            if ann_line == '':
                break

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
                 'end': str(end_sec), 'text': ann_line[ann_line.find('\t') + 1:].rstrip()})



            id_count += 1


# Similarity between our CLIP sentences  

if __name__ == '__main__':
    print('Start textual comparison')
    lsmdc_processor = LSMDCProcessor()
    lsmdc_processor.process_annotation_file()
