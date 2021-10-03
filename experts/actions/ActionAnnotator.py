import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from .stepwrapper.STEP.utils.vis_utils import draw_rectangle

# import from common
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.BaseAnnotator import BaseAnnotator


class ActionAnnotator(BaseAnnotator):
    def overlay_prediction_on_frame(self, frame, detections):
        """
        Expected detections: a dictionary of the form
        {
            'detection_boxes': a list of lists of floating points. each list is a box for a single
                                object in format xyxy, with values between 0 and 1.
            'detection_scores': a list of lists floating points. each list contains the confidence
                                for every predicted action class in the matching box region.
            'detection_classes': a list of lists of string action names, e.g. ["stand", "carry"], ["sing", "dance"],
                                 ["sit"], ...
        }
        """
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
                draw.text((box[0]+10, box[1]+10+count*20), f'{label}: {score:.2f}', fill="green")
                count += 1

        return np.array(I)[..., ::-1]
