import os
import pickle
import random
import requests
import shutil
import zipfile

from PIL import Image
import visual_genome.local as vg
import wget

from .DataSource import DataSource

DATA_DIR = os.path.join(os.path.dirname(__file__), 'vg_data/')
IMAGE_DATA_DIR = os.path.join(DATA_DIR, 'by-id/')
CACHE_DIR = os.path.join(DATA_DIR, 'cache/')
SYNSETS_FILE = os.path.join(DATA_DIR, 'synsets.json')

DATA_URL_TEMPLATE = 'http://visualgenome.org/static/data/dataset/{filename}'
DATA_FILES = ['image_data.json', 'scene_graphs.json', os.path.basename(SYNSETS_FILE)]

class VisualGenome(DataSource):
    def __init__(self):
        self._download_data_if_needed()
        self._parse_scene_graphs_if_needed()
        self.imgs_data = vg.get_all_image_data(DATA_DIR)

    def __len__(self):
        return len(self.imgs_data)

    def __getitem__(self, index):
        cached = self._check_cache(index)
        if cached:
            with open(cached, 'rb') as f:
                return pickle.load(f)

        else:
            data = self.imgs_data[index]
            img = Image.open(requests.get(data.url, stream=True).raw)
            sg = vg.get_scene_graph(data.id, images=DATA_DIR, image_data_dir=IMAGE_DATA_DIR,
                                    synset_file=SYNSETS_FILE)

            self._cache_pair((img, sg), index)

            return img, sg

    def _download_data_if_needed(self):
        os.makedirs(DATA_DIR, exist_ok=True)

        for data_file in DATA_FILES:
            
            if os.path.exists(os.path.join(DATA_DIR, data_file)):
                continue
            
            print(f'downloading data file {data_file}')
            self.__download_extract_delete(f'{data_file}.zip')
            print()

    def _parse_scene_graphs_if_needed(self):
        if not os.path.exists(IMAGE_DATA_DIR):
            print(f'\nparsing scene graphs. This may take a few minutes (this is done once) \n')
            vg.save_scene_graphs_by_id(data_dir=DATA_DIR, image_data_dir=IMAGE_DATA_DIR)
    
    @staticmethod
    def _delete_parsed_scene_graphs():
        if os.path.isdir(IMAGE_DATA_DIR):
            shutil.rmtree(IMAGE_DATA_DIR)
    
    @staticmethod
    def _delete_all_data():
        if os.path.isdir(DATA_DIR):
            shutil.rmtree(DATA_DIR)
    
    def _check_cache(self, index):
        cache_path = self.__get_cache_filename(index)
        if os.path.isfile(cache_path):
            return cache_path
        else:
            return None

    def _cache_pair(self, pair, index):
        cache_path = self.__get_cache_filename(index)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(pair, f)

    @staticmethod
    def __download_extract_delete(zip_name):
        # download
        zip_url = DATA_URL_TEMPLATE.format(filename=zip_name)
        download_path = os.path.join(DATA_DIR, zip_name)
        wget.download(zip_url, download_path)

        # extract
        with zipfile.ZipFile(download_path) as z:
            z.extractall(DATA_DIR)

        # delete
        os.remove(download_path)

    @staticmethod
    def __get_cache_filename(index):
        return os.path.join(CACHE_DIR, f'{index}.pkl')
