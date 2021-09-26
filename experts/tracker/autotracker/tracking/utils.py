import os
import sys
from queue import Queue

import cv2
from shapely.geometry import box
from tqdm import tqdm

# import from common
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from common.datasets import VideoFile, FrameImagesFolder

TRACKER_TYPE_CSRT = 'CSRT'
TRACKER_TYPE_KCF  = 'KCF'


class MultiTracker:
    """
    A wrapper for operating multiple trackers at once. Attempt to copy old OpenCV API:
    https://docs.opencv.org/4.5.0/d8/d77/classcv_1_1MultiTracker.html
    """

    def __init__(self, default_tracker: str = TRACKER_TYPE_CSRT):
        """
        Create new multitracker object
        @param: default_tracker: the default tracker to add with `add_tracker` function. Use `TRACKER_TYPE_` constants
        """
        self.id_counter = 0
        self.trackers = {}
        self.latest_tracker_data = {}
        self.default_tracker = default_tracker

    def add_tracker(self, frame, initial_bbox, score, cls, tracker_type: str = None):
        """
        Add a new object tracker.
        @param: frame: the frame in which the object was detected.
        @param: initial_box: the bounding box of the detected object (format xywh)
        """

        if not tracker_type:
            tracker_type = self.default_tracker

        # create new tracker for object in box on frame
        new_tracker = getattr(cv2, f'Tracker{tracker_type}_create')()
        new_tracker.init(frame, initial_bbox)

        # save under next id
        self.trackers[self.id_counter] = new_tracker
        self.latest_tracker_data.setdefault(self.id_counter, {})['box'] = initial_bbox
        self.latest_tracker_data[self.id_counter]['score'] = score
        self.latest_tracker_data[self.id_counter]['class'] = cls

        self.id_counter += 1

    def update(self, frame):
        """
        Update all trackers according to the next video frame
        @param: frame: the next video frame. A numpy array of shape (H, W, C).
        @return: a tuple (removed, updated):
                 removed - a list of ID's of objects that are no longer being tracked
                 updated - a dictionary, obj_id --> box (format xywh), for all boxes still being tracked.
        """

        # update each tracker individually and save results
        update_res = {}
        for i, tracker in self.trackers.items():
            update_res[i], self.latest_tracker_data[i]['box'] = tracker.update(frame)

        # if object is not visible (update results "False") remove tracker.
        removed_trackers = []
        for i, res in update_res.items():
            if not res:
                self.latest_tracker_data.pop(i)
                self.trackers.pop(i)
                removed_trackers.append(i)
        
        return removed_trackers, {i: self.latest_tracker_data[i]['box'] for i in self.latest_tracker_data}

    def merge_new_detections(self, frame, detection_boxes, detection_scores, detection_classes,
                             iou_thresh=0.95, tracker_type: str = None, remove_unseen_by_model=False):
        """
        merge new detections to currently tracked objects according to IOU score.
        @param: frame: the detection frame. A numpy array of shape (H, W, C).
        @param: detection_boxes: a list of detected item boxes in format xywh.
        @param: detection_scores: a list of confidence probabilities for each detected object.
        @param: detection_classes: a list of string class names for each detected object.
        @param: iou_thresh: an IOU threshold for object merging.
        @param: tracker_type: Tracker type for newly added trackers. Use `TRACKER_TYPE_` constants.
        @param: remove_unseen_by_model: If true, tracked objects that are not detected by the model will be removed from the tracker.
        """
        # if no new detections, skip function
        if len(detection_boxes) == 0:
            return

        # set tracker type if not given
        if not tracker_type:
            tracker_type = self.default_tracker

        found_matches = set()
        trackers_to_add = []
        for bbox, score, cls in zip(detection_boxes, detection_scores, detection_classes):
            match_box = self.find_close_box_id(bbox, cls, iou_thresh=iou_thresh)

            if match_box >= 0:  # found match. continue with tracked box
                found_matches.add(match_box)
                self.latest_tracker_data[match_box]['score'] = score  # update confidence score
                continue
            else:  # not matching tracked box. add new tracker
                bbox = tuple(round(b) for b in bbox)
                trackers_to_add.append((frame, bbox, score, cls, tracker_type))

        # add trackers after iteration to avoid removing close objects
        for tracker_args in trackers_to_add:
            found_matches.add(self.id_counter)
            self.add_tracker(*tracker_args)

        # delete trackers with no matching detection
        if remove_unseen_by_model:
            remove_trackers = [track_id
                            for track_id in self.latest_tracker_data
                            if track_id not in found_matches]
            for track_id in remove_trackers:
                self.latest_tracker_data.pop(track_id)
                self.trackers.pop(track_id)

    def find_close_box_id(self, detection_box, detection_class, iou_thresh=0.95):
        """
        find a tracked box that is close (in terms of IOU score) to a new detection box.
        @param: detection_box: a bounding box (format xywh).
        @param: detection_class: the string class name of the detected object. corresponding to the box.
        @param: iou_thres: an IOU threshold for object merging.
        @return: The ID of the object that has the highest IOU score with the given detection box. returns -1
                 if no match is found with a score above `iou_thresh`.
        """
        max_score = 0
        match_id = -1
        for i, tracked_data in self.latest_tracker_data.items():
            if detection_class != tracked_data['class']:  # match boxes only if classes are the same
                continue

            score = self.__iou_score(tracked_data['box'], detection_box)
            if score > max_score:
                max_score = score
                match_id = i
        
        if max_score > iou_thresh:
            return match_id
        else:
            return -1

    @staticmethod
    def __iou_score(box1, box2):
        """
        Calculates the IOU score between two bounding boxes.
        @param box1: a bounding box in format xywh.
        @param box2: a bounding box in format xywh.
        @return: The IOU score between the two boxes.
        """
        # convert boxes to xyxy format
        (x, y, w, h) = box1
        box1 = (x, y, x+w, y+h)
        
        (x, y, w, h) = box2
        box2 = (x, y, x+w, y+h)

        # create polygon objects
        poly1 = box(*box1)
        poly2 = box(*box2)

        # claculate intersection and union
        poly_intersection = poly1.intersection(poly2).area
        poly_union = poly1.union(poly2).area

        return poly_intersection / poly_union

    @classmethod
    def track_video_objects(cls, video_path, detection_model, detect_every=10,
                            merge_iou_threshold=0.95, tracker_type: str = TRACKER_TYPE_KCF,
                            refresh_on_detect=True, show_pbar=True, logger=None):
        """
        @param: video_path: string, Video Full Path.
        @param: detection_model: object detection model or detections queue.
                                 If a model is given, it must implement `predict_single_frame` with the uniform API
        @param: detect_every: perform object detection with `detection_model` every `detect_every` frames
        """
        # load dataset
        if os.path.isdir(video_path):
            ds = FrameImagesFolder(video_path)
        else:
            ds = VideoFile(video_path)

        # output aggregator
        track_data = {}

        # init multi-tracker object
        multitracker = cls(default_tracker=tracker_type)

        # iterate video frames
        for frame_count, frame in tqdm(enumerate(ds), total=len(ds), disable=not show_pbar):
            
            # update all existing trackers to current frame
            multitracker.update(frame)

            # detect if necessary
            if frame_count % detect_every == 0:
                if isinstance(detection_model, Queue):  # if queue, get next predictions
                    detections = detection_model.get()
                    if detections == 'STOP':
                        # if we get the STOP command, stop everything and raise an early stopping exception.
                        # skip save and skip any furter tracking
                        raise EarlyStopException()
                else:  # if detection model, perform detection
                    detections = detection_model.predict_single_frame(frame)

                # log new detections
                if logger:
                    logger.info(f'tracking {video_path}: got frame {frame_count}/{ds.num_frames} detections')

                # merge new detections with existing
                multitracker.merge_new_detections(frame, **detections, iou_thresh=merge_iou_threshold,
                                                  remove_unseen_by_model=refresh_on_detect)

            # save existing detections for current frame.
            for i, data in multitracker.latest_tracker_data.items():
                frame_id, obj_id = map(str, [frame_count, i])

                if obj_id not in track_data:
                    track_data[obj_id] = {
                        'class': data['class'],
                        'scores': {},
                        'boxes': {}
                    }

                (x, y, w, h) = data['box']
                track_data[obj_id]['boxes'][frame_id] = [x, y, x+w, y+h]
                track_data[obj_id]['scores'][frame_id] = float(data['score'])

            frame_count += 1

        del ds

        # add start,stop frames as info
        for obj_data in track_data.values():
            obj_frames_as_int = list(map(int, obj_data['boxes'].keys()))
            obj_data.update({
                'start_frame': min(obj_frames_as_int),
                'stop_frame': max(obj_frames_as_int) + 1,
            })

        return track_data


class EarlyStopException(Exception):
    pass
