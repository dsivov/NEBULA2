import copy
import json
import os
import sys

import cv2
from tqdm import tqdm

from autotracker import VideoDataset, FramesDataset


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
    if os.path.isdir(video_path):
        ds = FramesDataset(video_path)
        frames_per_second = 30  # TODO handle 30 FPS default
    else:
        ds = VideoDataset(video_path)
        frames_per_second = ds.video.get(cv2.CAP_PROP_FPS)

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

    # change annotations mapping format for easier frames iteration.
    # change to frame --> obj --> box format.
    anno_fromatted = {}
    for obj_id, obj_data in annotations.items():
        for frame_id, box in obj_data['boxes'].items():
            anno_fromatted.setdefault(frame_id, {}).setdefault(obj_id, {})['box'] = box
            anno_fromatted[frame_id][obj_id]['score'] = obj_data['scores'][frame_id]
            anno_fromatted[frame_id][obj_id]['class'] = obj_data['class']
    annotations = anno_fromatted

    # iterate all frames
    for i, frame in tqdm(enumerate(ds), total=len(ds), desc=f'annotating {video_path}', disable=not show_pbar):

        # assert string ID
        frame_str_id = str(i)
        
        if frame_str_id not in annotations:
            merged_frame = frame  # no objects found in this frame. use original frame
        else:
            # mark all objects for this frame in a new copy.
            merged_frame = merge_prediction_to_frame(frame, annotations[frame_str_id])

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
    # make a copy of the frame so as not to change the original
    new_frame = copy.deepcopy(frame)

    # calculate font scale required for this video's annotations according to resolution.
    height, width, _ = new_frame.shape
    font_scale = 0.5 * width / 1280 + 0.5 * height / 640

    # iterate all detections
    for item_id, item_data in detections.items():
        
        # get detection info
        box = item_data['box']
        score = item_data['score']
        cls = item_data['class']

        # annotate frame with rectangle and text
        cv2.rectangle(new_frame, box[:2], box[2:], (0, 0, 255), 2)
        cv2.putText(new_frame, f'{item_id}:{cls}({int(score*100)}%)',
                    box[:2], cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 255, 0))

    return new_frame


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
