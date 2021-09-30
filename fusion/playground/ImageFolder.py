import os

from .DataSource import DataSource


class ImageDirectory(DataSource):
    def __init__(self, dir_path):
        super().__init__()
        self.dir_path = dir_path
        self.img_filenames = sorted(os.listdir(self.dir_path))

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        pass #TODO


