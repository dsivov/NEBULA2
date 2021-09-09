import os
import re

from abc import ABC, abstractmethod

import cv2
import numpy as np

from PIL import Image


class DetectionDataset(ABC):
    def __init__(self, frames_path, get_every=1, resize=None):
        self.frames_dir = frames_path
        self.get_every = get_every
        self.num_frames = self.initialize_frames()
        self.resize = resize
    
    @abstractmethod
    def initialize_frames(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def __len__(self):
        return -(-self.num_frames // self.get_every)  # equivalent to `ceil(num_frames / get_every)`


class VideoDataset(DetectionDataset):
    def initialize_frames(self):
        self.video = cv2.VideoCapture(self.frames_dir)
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        gen_count = 0
        while self.video.isOpened():
            self.video.set(cv2.CAP_PROP_POS_FRAMES,(gen_count*self.get_every))
            success, frame = self.video.read()
            if success:
                if self.resize:
                    frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_CUBIC)
                yield frame
            else:
                break

            gen_count += 1

    def __del__(self):
        self.video.release()


class FramesDataset(DetectionDataset):
    def initialize_frames(self):
        self.frames =  sorted(list(filter(
            lambda fname: re.match(r'^frame\d{4}.jpg$', fname),
            os.listdir(self.frames_dir)
        )))

        return len(self.frames)
    
    def __iter__(self):
        for i in range(0, self.num_frames, self.get_every):
            img = Image.open(os.path.join(self.frames_dir, self.frames[i]))
            if self.resize:
                img = img.resize(self.resize, resample=Image.BICUBIC)
            yield np.array(img)[:, :, ::-1]
