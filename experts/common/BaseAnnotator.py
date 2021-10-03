import os
from abc import ABC, abstractmethod

import cv2
import json
from tqdm import tqdm

from .datasets import FrameImagesFolder, VideoFile


class BaseAnnotator(ABC):
    @abstractmethod
    def overlay_prediction_on_frame(self, frame, detections):
        """
        add all detections to their corresponding frame.
        @frame: a numpy array of shape (H, W, C) in color format BGR.0
        @detections: a detection object.
        @return: a new frame that is identical to the given frame with the given detections annotated on top.
        """
        pass

    def annotate_video(self, video_path, annotations, output_path, video_fps=25, show_pbar=True):
        """
        annotate an entire video.
        @param: video_path: path to the video container (video file or frames directory)
        @param: annotations: a dicitonary in the form:
                {
                    "0": detections in frame 0

                    "1": detections in frame 1
                    "2": ...
                    ...
                }
                The actual detections object is dependant on the inheritor's implementation of
                the `overlay_prediction_on_frame` function.
        @param: output_path: The path to save the output annotated video (without extension).
        @param: show_pbar: If True, a progress bar is shown while annotating.
        """
        
        # load dataset
        if os.path.isdir(video_path):
            ds = FrameImagesFolder(video_path)
        else:
            ds = VideoFile(video_path)

        # create video writer object
        height, width, _ = next(iter(ds)).shape
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)  # create path if necessary.
        output_file = cv2.VideoWriter(
            filename=f'{output_path}.mp4',
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=float(video_fps),
            frameSize=(width, height),
            isColor=True
        )

        # iterate all frames
        for i, frame in tqdm(enumerate(ds), total=len(ds), desc=f'annotating {video_path}', disable=not show_pbar):

            # assert string ID
            frame_str_id = str(i)
            
            if frame_str_id not in annotations:
                merged_frame = frame  # no objects found in this frame. use original frame
            else:
                # mark all objects for this frame in a new copy.
                merged_frame = self.overlay_prediction_on_frame(frame, annotations[frame_str_id])

            # write annotated frame to video
            output_file.write(merged_frame)

        # release everything
        del ds
        output_file.release()

    def annotate_video_from_json(self, video_path, annotation_path, output_path, video_fps=25, show_pbar=True):
        """
        Annotate a video with annotations in a JSON file.
        @param: video_path: path to the video container (video file or frames directory).
        @param: annotation_path: a path to a JSON file containing object data in the correct format (see
                documentation for `annotate_video` function).
        @param: output_path: The path to save the output annotated video (without extension).
        """
        # open annotations file and read
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        # annotate video with opened annotations
        return self.annotate_video(video_path, annotations, output_path, video_fps, show_pbar)
