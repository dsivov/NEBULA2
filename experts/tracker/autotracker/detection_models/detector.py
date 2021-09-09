from abc import ABC, abstractmethod
import os
from queue import Queue

import numpy as np
from tqdm import tqdm

from ..data.datasets import VideoDataset, FramesDataset

class PretrainedVideoPredictor(ABC):
    """
    A video wrapper for the tensorflow object detection API.
    """

    def __init__(self, pretrained_model_cfg: str, confidence_threshold: float = 0.5):
        """
        Create a predictor of the given pretrained model config.
        @param: pretrained_model_cfg: the name of a model config file. use CFG_ constants.
        @param: confidence_threshold: only save predictions with a confidence score above this threshold.
        """
        print('loading model')
        self.model_cfg = pretrained_model_cfg
        self.confidence_t = confidence_threshold
        self.model, self.label_map = self.load_model()

    @abstractmethod
    def load_model(self):
        """
        load a model with pretrained weights and accompanying label map.
        @return: a tuple (model, label_map)
        """
        pass

    @abstractmethod
    def predict_batch(self, image_batch: np.ndarray):
        """
        Perform prediciton on a batch of images.
        @param: image_batch: a numpy array of shape (batch_size, H, W, C)
        """
        pass

    def predict_video(self,
                      path_to_video,
                      batch_size=8,
                      pred_every=1,
                      show_pbar=True,
                      global_aggregator=None):
        """
        Performs a prediciton on every frame of a video.
        @param: path_to_video: absolute or relative path to a video file.
        @param: batch_size: the number of images on which to do inference simultaneously.
        @param: pred_every: how many frames to skip between predicitons.
        @param: show_pbar: if True, a progressbar is shown while working on a video.
        @param: global_aggregator: Any aggregator that implements append or extend, or a python Queue object.
                If a queue is given, predictions are saved as dicitonaries of the form:
                {
                    'video_path': path to video in detection,
                    'pred': prediction dictionary
                }
                If None, create a local list and return at the end.
        @return: a list of prediction dictionaries in format:
                 {
                     "detection_boxes": (x, y, w, h),
                     "detection_classes": the string name of the predicted class,
                     "detection_scores": prediction confidence probability in [0, 1]
                 }
                 If a queue was given as a global aggregator, the queue is returned in the format
                 described in the parameter description.
        """
        # load dataset
        if os.path.isdir(path_to_video):
            ds = FramesDataset(path_to_video, get_every=pred_every)
        else:
            ds = VideoDataset(path_to_video, get_every=pred_every)

        # save predictions in global aggregator if provided
        if global_aggregator is not None:
            preds = global_aggregator
        else:
            preds = []


        # iterate video frames for predicitons
        pbar = tqdm(total=len(ds), disable=not show_pbar)
        input_iter = iter(ds)
        done = False
        while not done:

            # aggregate batch of frames
            batch = []
            for _ in range(batch_size):
                try:
                    inp = next(input_iter)
                    batch.append(inp)
                except StopIteration:
                    done = True

            # perform prediction on batch
            if batch:
                batch_preds = self.predict_batch(np.stack(batch)) 
                preds.extend(batch_preds)
                
                pbar.update(len(batch))
        
        pbar.close()
        del ds

        return preds

    def predict_single_frame(self, frame):
        """
        Perform prediction on a single video frame.
        @param: frame: a numpy array of sape (H, W, C)
        @param: uniform_api: if True, returns a uniform API prediction dictionary:
                             {
                                 "detection_boxes": (x, y, w, h),
                                 "detection_classes": the string name of the predicted class,
                                 "detection_scores": prediction confidence probability in [0, 1]
                             }
        @return: a tensorflow object detection API output dictionary or uniform API dictionary.
        """
        # predict single frame as a batch of size 1
        return self.predict_batch(np.stack([frame]))[0]
