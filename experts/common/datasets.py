import os
import re

from abc import ABC, abstractmethod

import cv2
import numpy as np

from PIL import Image

from .RemoteAPIUtility import RemoteAPIUtility


class FramesDataset(ABC):
    """
    A superclass for datasets used for video frame object detection
    """

    def __init__(self, frames_path, get_every=1, resize=None):
        """
        Create a new dataset according to frames source.
        @param: frames_path: relative or abosolute path to the location of the frames. Depends on the
                             concrete DetectionDataset type.
        @param: get_every: Determines the "active" frames in this dataset, i.e. iterating the dataset will
                           yield every `get_every` frame in the dataset by sequential order.
        @param: resize: A tuple (width, height) determining how to resize the frame when loading. If None,
                        the original frame size is used.
        """
        self.frames_dir = frames_path
        self.get_every = get_every
        self.num_frames = self.initialize_frames()
        self.resize = resize
    
    @abstractmethod
    def initialize_frames(self):
        """
        Initialize the dataset so that it will be ready to iterate.
        @return: the number of frames in `frames_dir`.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Iterates over all desired frames in the dataset.
        @return: yields each frame of the dataset sequentially. output frame format is BGR.
        """
        pass

    def __len__(self):
        """
        Calculates number of utilised frames in the dataset.
        @return: the number of utilised frames in the dataset.
        """
        return -(-self.num_frames // self.get_every)  # equivalent to `ceil(num_frames / get_every)`


class VideoFile(FramesDataset):
    """
    A `DetectionDataset` for video file frame source.
    """

    def initialize_frames(self):
        self.video = cv2.VideoCapture(self.frames_dir)  # open and keep video object
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))  # get frame count from video object

    def __iter__(self):
        gen_count = 0
        while self.video.isOpened():
            self.video.set(cv2.CAP_PROP_POS_FRAMES,(gen_count*self.get_every))  # set video cursor to next desired frame
            success, frame = self.video.read()  # read next frame
            if success:
                if self.resize:  # resize if needed
                    frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_CUBIC)
                yield frame
            else:
                break  # out of frames

            gen_count += 1

    def __del__(self):
        self.video.release()  # must release the video file (closes the file stream)


class FrameImagesFolder(FramesDataset):
    """
    A `DetectionDataset` for directory of frames source. frames must have names of the form "frameXXXX.jpg"
    where XXXX is some number written in 4 digits. These frames are expected to be sequential by their order
    in the original video.
    """

    def initialize_frames(self):
        # save file names of frames in the directory in sequential order.
        # imp: filter files according to frame file name format, then sort.
        self.frames =  sorted(list(filter(
            lambda fname: re.match(r'^frame\d{4}.jpg$', fname),
            os.listdir(self.frames_dir)
        )))

        return len(self.frames)
    
    def __iter__(self):
        # iterate indices of desired frames
        for i in range(0, self.num_frames, self.get_every):

            # open image and resize
            img = Image.open(os.path.join(self.frames_dir, self.frames[i]))
            if self.resize:
                img = img.resize(self.resize, resample=Image.BICUBIC)

            # return as BGR image (like cv2 video).
            yield np.array(img)[:, :, ::-1]


class RemoteVideo(FrameImagesFolder):
    def initialize_frames(self):
        remote = RemoteAPIUtility()

        # frames_dir should be arango_id, e.g. Movies/12345678
        # download movie and continue as images forlder.
        num_frames_downloaded = remote.downloadDirectoryFroms3(self.frames_dir)
        if num_frames_downloaded == 0:
            raise ValueError(f'no frames to download at ID {self.frames_dir}')

        return super().initialize_frames()
