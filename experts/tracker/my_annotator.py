import copy
import json
import os
import tempfile

import cv2
import numpy as np
from tqdm import tqdm

from autotracker import VideoDataset, FramesDataset


def annotate_video(video_path, annotations, output_path, show_pbar=True):
    # load dataset
    if os.path.isdir(video_path):
        ds = FramesDataset(video_path)
        frames_per_second = 30  # TODO handle 30 FPS default
    else:
        ds = VideoDataset(video_path)
        frames_per_second = ds.video.get(cv2.CAP_PROP_FPS)
    height, width, _ = next(iter(ds)).shape

    codec, file_ext = (
        ("x264", ".mkv") if __test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    )

    output_file = cv2.VideoWriter(
        filename=f'{output_path}{file_ext}',
        # some installation of opencv may not support x264 (due to its license),
        # you can try other format (e.g. MPEG)
        fourcc=cv2.VideoWriter_fourcc(*codec),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True
    )

    # annotations to frame --> obj --> box format
    anno_fromatted = {}
    for obj_id, obj_data in annotations.items():
        for frame_id, box in obj_data['boxes'].items():
            anno_fromatted.setdefault(frame_id, {}).setdefault(obj_id, {})['box'] = box
            anno_fromatted[frame_id][obj_id]['score'] = obj_data['scores'][frame_id]
            anno_fromatted[frame_id][obj_id]['class'] = obj_data['class']
    annotations = anno_fromatted

    for i, frame in tqdm(enumerate(ds), total=len(ds), desc=f'annotating {video_path}', disable=not show_pbar):
        frame_str_id = str(i)
        if frame_str_id not in annotations:
            merged_frame = frame
        else:
            merged_frame = merge_prediction_to_frame(frame, annotations[frame_str_id])

        output_file.write(merged_frame)

    del ds
    output_file.release()


def merge_prediction_to_frame(frame, detections):
    new_frame = copy.deepcopy(frame)

    for item_id, item_data in detections.items():
        box = item_data['box']
        score = item_data['score']
        cls = item_data['class']
        cv2.rectangle(new_frame, box[:2], box[2:], (0, 0, 255), 2)
        cv2.putText(new_frame, f'{item_id}:{cls}({int(score*100)}%)',
                    box[:2], cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0))

    return new_frame


def annotate_video_from_json(video_path, annotation_path, output_path):
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    return annotate_video(video_path, annotations, output_path)


def frames_generator(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


# from: detectron2/demo/demo.py
def __test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    annotate_video('/movies/scenecliptest00581.avi', '/home/guy/nebula/src/engines/my_anno.json', 'my_out')
