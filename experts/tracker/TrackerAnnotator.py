import copy
import os
import sys

import cv2

# import from common
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.BaseAnnotator import BaseAnnotator


class TrackerAnnotator(BaseAnnotator):
    def annotate_video(self, video_path, annotations, output_path, video_fps=25, show_pbar=True):
        # change annotations mapping format for easier frames iteration.
        # change to frame --> obj --> box format.
        anno_fromatted = {}
        for obj_id, obj_data in annotations.items():
            for frame_id, box in obj_data['boxes'].items():
                anno_fromatted.setdefault(frame_id, {}).setdefault(obj_id, {})['box'] = box
                anno_fromatted[frame_id][obj_id]['score'] = obj_data['scores'][frame_id]
                anno_fromatted[frame_id][obj_id]['class'] = obj_data['class']
        annotations = anno_fromatted

        return super().annotate_video(video_path, annotations, output_path, video_fps=video_fps, show_pbar=show_pbar)

    def overlay_prediction_on_frame(self, frame, detections):
        """
        Expected detections: a dictionary of the form
        {
            'detection_boxes': a list of lists of floating points. each list is a box for a single
                                object in format xyxy, with values between 0 and 1.
            'detection_scores': a list floating points. each number is the confidence for every predicted category class
                                in the matching box region.
            'detection_classes': a list of lists of string action names, e.g. ["stand", "carry"], ["sing", "dance"],
                                 ["sit"], ...
        }
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
