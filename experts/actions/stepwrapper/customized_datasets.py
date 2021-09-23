from genericpath import exists
import os
import glob

import cv2
from .STEP.data.customize import CustomizedDataset as CustomizedMultiMovieFolder

class CustomizedFrameImagesFolder(CustomizedMultiMovieFolder):
    def make_list(self):
        video_name = os.path.basename(self.data_root)
        self.data = []
        frames = glob.glob(self.data_root + '/*.jpg')
        self.data_root = os.path.dirname(self.data_root)
        numf = len(frames)
        for i in range(numf):
            self.data.append((video_name, i, numf))


class CustomizedVideoFile(CustomizedFrameImagesFolder):
    def __init__(self, data_root, T=3, chunks=3, source_fps=30, target_fps=12, transform=None, stride=1,
                 anchor_mode="1", im_format='frame%04d.jpg'):
        frames_out = f'Movies/{os.path.splitext(os.path.basename(data_root))[0]}'
        os.makedirs(frames_out, exist_ok=True)
        print(os.path.join(os.getcwd(), frames_out))
        self.extractImages(data_root, frames_out)
        super().__init__(frames_out, T=T, chunks=chunks, source_fps=source_fps, target_fps=target_fps,
        transform=transform, stride=stride, anchor_mode=anchor_mode, im_format=im_format)
    
    @staticmethod
    def extractImages(pathIn, pathOut):
        count = 0
        vidcap = cv2.VideoCapture(pathIn)
        success,image = vidcap.read()
        success = True
        while success:
            success,image = vidcap.read()
            if success:
                cv2.imwrite(os.path.join(pathOut,'frame%04d.jpg'%count), image)
            count = count + 1
