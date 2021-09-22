import copy
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from action_detector.STEP.utils.vis_utils import draw_rectangle

from datasets import VideoDataset, FramesDataset


def annotate_video(video_path, annotations, output_path, show_pbar=True):
    """
    annotate an entire video.
    @param: video_path: path to the video container (video file or frames directory)
    @param: annotations: a dicitonary in the form:
            {
                "0" (object ID): {
                    "class": class_name (string)
                    "scores": {
                        frame_num (as string): confidence score in this frame
                        frame_num: confidence score in this frame
                        ...
                    }
                    "boxes" {
                        frame_num (as string): box in this frame [x_min, y_min, x_max, y_max]
                        frame_num: [x_min, y_min, x_max, y_max]
                        ...
                    }
                    "start_frame": first frame the object appears
                    "stop_frame": first frame where the object does not appear anymore
                }

                "1": ...
                "2": ...
            }
    @param: output_path: The path to save the output annotated video (without extension).
    @param: show_pbar: If True, a progress bar is shown while annotating.
    """
    
    # load dataset
    ds = FramesDataset(video_path)
    frames_per_second = 30  # TODO handle 30 FPS default

    # create video writer object
    height, width, _ = next(iter(ds)).shape
    output_file = cv2.VideoWriter(
        filename=f'{output_path}.mp4',
        # some installation of opencv may not support x264 (due to its license),
        # you can try other format (e.g. MPEG)
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True
    )

    # iterate all frames
    for frame, anno in zip(tqdm(ds,
                                total=len(ds),
                                desc=f'annotating {video_path}',
                                disable=not show_pbar), annotations):

        # mark all objects for this frame in a new copy.
        merged_frame = merge_prediction_to_frame(frame, anno)

        # write annotated frame to video
        output_file.write(merged_frame)

    # release everything
    del ds
    output_file.release()


def merge_prediction_to_frame(frame, detections):
    """
    add all detections to their corresponding frame.
    @frame: a numpy array of shape (H, W, C) in color format BGR.
    @detections: a dictionary of objects, each with a box, score and class for that object as it appears in
                 the current frame.
    @return: a new frame that is identical to the given frame with the given detections annotated on top.
    """
    # calculate font scale required for this video's annotations according to resolution.
    I = Image.fromarray(frame[..., ::-1])
    W, H = I.size

    draw = ImageDraw.Draw(I)

    # iterate all detections
    merged_result = {}
    for box, classes, scores in zip(detections["detection_boxes"],
                                    detections["detection_classes"],
                                    detections["detection_scores"]):
        merged_result[','.join(map(str, box))] = [(c, s) for c, s in zip(classes, scores)]

        box = np.asarray(box, dtype=np.float32)
        box[::2] *= W
        box[1::2] *= H
        draw_rectangle(draw, box, outline="red")

        count = 0
        for label, score in zip(classes, scores):
            draw.text((box[0]+10, box[1]+10+count*20), f'{label}: {score:.2f}', fill="white")
            count += 1

    return np.array(I)[..., ::-1]


def annotate_video_from_json(video_path, annotation_path, output_path):
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
    return annotate_video(video_path, annotations, output_path)


if __name__ == "__main__":
    annotate_video_from_json(video_path=sys.argv[1],
                             annotation_path=sys.argv[2],
                             output_path=sys.argv[3] if len(sys.argv) >= 4 else 'annotated')
