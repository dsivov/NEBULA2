import argparse
import json
import logging
import os
import sys

from datetime import datetime
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import tracker.autotracker as at
from tracker.TrackerAPIUtility import TrackerAPIUtility
from tracker.TrackerAnnotator import TrackerAnnotator

# import from common
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.ExpertManager import (ExpertManager, ExpertPipeline, ExpertPipelineStep, CLI_command, global_config,
                                  AggQueue, OUTPUT_STYLE_ANNO, OUTPUT_STYLE_JSON, OUTPUT_STYLE_ARANGO)
from common.video_util import VideoInfo

# ===== Input Constants =====
# default model configurations
CFG_DEFAULT = {
    at.BACKEND_TFLOW: 'CFG_OID_V4_DETECTION_FerRCNN_INCEPTION_V2',
    at.BACKEND_DETECTRON: 'CFG_COCO_DETECTION_FerRCNN_X101_32x8d_FPN_LR3x'
}

# ===== Command Constants =====
# special queue messages for event control
TRACK_COMPLETE_MSG = 'TRACK COMPLETE'
TRACK_STOPPED_MSG = 'TRACK STOPPED'
TRACK_FAILED_MSG = 'TRACK FAILED'

# task message keys
VIDEO_PATH_KEY = 'video_path'
IS_REMOTE_KEY = 'is_remote'
PREDICTION_KEY = 'predictions'
FPS_KEY = 'fps'
NUM_FRAMES_KEY = 'num_frames'

# step names
OBJECT_DETECTOR = 'Object Detector'
TRACKER_DISPATCHER = 'Tracker Dispatcher'
SCHEDULER = 'Scheduler'

class TrackerManager(ExpertManager):

    def initialize(self):
        self.api = TrackerAPIUtility()
        self.cur_task = None
        self.model = self.__load_model()

        self.tracker_dispatch_dict = {}

    def get_pipeline(self) -> ExpertPipeline:
        detector_step = DetectorStep(OBJECT_DETECTOR)
        tracker_step = TrackerStep(TRACKER_DISPATCHER)
        if self.args.arango:
            arango_step = ArangoStep('Scheduler', is_daemon=True)
            return ExpertPipeline([(arango_step, detector_step),
                                   (detector_step, tracker_step)])
        else:
            return ExpertPipeline([(detector_step, tracker_step)])

    def __load_model(self):
        if self.args.backend is None and self.args.model is None:
            self.args.backend = at.active_detection_backend()
            self.args.model = CFG_DEFAULT[self.args.backend]

        # no config chosen. use default config for given backend
        elif self.args.model is None:
            at.set_active_backend(self.args.backend)
            self.args.model = CFG_DEFAULT[self.args.backend]
        
        # no backend chosen. find backend that has given config
        elif self.args.backend is None:
            possible_backends = at.backends_with_config(self.args.model)
            if not possible_backends:
                raise ValueError(f'Model configuration {self.args.model} not found in any backend available in this evironment')
            self.args.backend = possible_backends[0]
            at.set_active_backend(self.args.backend)

        # chosen backend and config. check compatibility
        else:
            at.set_active_backend(self.args.backend)
            if not hasattr(at.detection_utils, self.args.model):
                raise ValueError(f'Given model backend and config are incopatible in the current environment: {self.args.backend} - {self.args.model}')

        return at.detection_utils.VideoPredictor(getattr(at.detection_utils, self.args.model),
                                                         confidence_threshold=self.args.confidence)

    @CLI_command
    def local(self, line):
        """run action detection on local movie file or frames directory, e.g.: local /movies/scenecliptest00581.avi"""
        if os.path.exists(line):
            detector_q = self.pipeline.incoming_queues[OBJECT_DETECTOR]
            try:
                detector_q.put(self._get_video_msg(line, is_remote=False))
            except:
                self.logger.exception(f'bad video path "{line}"')
        else:
            self.print_n_log(f'path {line} does not exist')

    @CLI_command
    def remote(self, line):
        """run action detection on remote movie file, e.g.: remote Movies/92354428"""
        try:
            self.api.get_movie_info(line)  # check for movie info. if no error then movie exists
            detector_q = self.pipeline.incoming_queues[OBJECT_DETECTOR]
            detector_q.put(self._get_video_msg(line, is_remote=True))
        except ValueError:
            self.logger.exception(f'remote movie does not exist')

    @CLI_command
    def tasks(self, line=''):
        """view enqueued tasks"""
        detector_q = self.pipeline.incoming_queues[OBJECT_DETECTOR]
        self.print_n_log(f'detection tasks queue: {list(detector_q.queue)}')
        self.print_n_log(f'detection task in progress: {self.cur_task}')

        dispatcher_queue = self.pipeline.incoming_queues[TRACKER_DISPATCHER]
        self.print_n_log(f'detection tasks queue: {list(dispatcher_queue.queue)}')
        self.print_n_log(f'detection task in progress: {list(self.tracker_dispatch_dict.keys())}')


    def _get_video_msg(self, video_path, is_remote):
        msg = {VIDEO_PATH_KEY: video_path, IS_REMOTE_KEY: is_remote}
        msg.update(self.get_current_config())

        video_info = VideoInfo(video_path)
        msg[FPS_KEY] = video_info.fps if video_info.fps else self.default_fps.get()
        msg[NUM_FRAMES_KEY] = video_info.num_frames

        return msg

    class pred_every(global_config):
        def __init__(self, default_value=10):
            super().__init__(default_value)

        def set(self, new_value: str):
            new_value_int = int(new_value)
            assert new_value_int > 0
            self._value = new_value_int
    
    class batch_size(global_config):
        def __init__(self, default_value=8):
            super().__init__(default_value)

        def set(self, new_value: str):
            new_value_int = int(new_value)
            assert new_value_int > 0
            self._value = new_value_int

    class refresh_on_detect(global_config):
        def __init__(self, default_value=True):
            super().__init__(default_value)

        def set(self, new_value: str):
            new_value_bool = eval(new_value)
            assert isinstance(new_value_bool, bool)
            self._value = new_value_bool

    class merge_iou_thresh(global_config):
        def __init__(self, default_value=0.5):
            super().__init__(default_value)

        def set(self, new_value: str):
            new_value_float = float(new_value)
            assert 0 <= new_value_float <= 1
            self._value = new_value_float

    class tracker_type(global_config):
        def __init__(self, default_value=at.tracking_utils.TRACKER_TYPE_KCF):
            super().__init__(default_value)

        def set(self, new_value: str):
            new_value_type = getattr(at.tracking_utils, f'TRACKER_TYPE_{new_value.upper()}')
            self._value = new_value_type

    


class DetectorStep(ExpertPipelineStep):
    def run(self, q_in: Queue, q_out: AggQueue):
        """
        Detection model thread code.
        @param: q_in: input queue for paths to videos on which to detect.
        @param: q_out: queue in which to place predictions
        @param: model: the detection model. must implement `predict_video` function
        """
        self.logger.info('detection thread ready for input')

        # global indicators
        global exit_all  # exit sequence flag
        global cur_detection_task  # current video in detection

        # iterate until STOP message
        for msg in iter(q_in.get, self.pipeline.STOP_MSG):
            # set current video path
            video_path = msg[VIDEO_PATH_KEY]
            cur_detection_task = video_path

            # download if necessary
            if msg[IS_REMOTE_KEY]:
                # download frames or continue if it does not exist
                self.logger.info(f'downloading remote movie: {video_path}')
                num_frames = self.api.downloadDirectoryFroms3(video_path)
                if num_frames == 0:
                    self.logger.error(f'no frames found under the name {video_path}')
                    cur_detection_task = None  # if no frames are found, empty current task indicator
                    continue

            self.logger.info(f'starting detection: {video_path}')

            # wrap queues to have the extend and append functions.
            # we save in the same message but template but add predicitons
            class QWrap:
                def __init__(self, q):
                    self.q = q

                def append(self, item):
                    new_msg = msg.copy()  # copy original message
                    new_msg[PREDICTION_KEY] = item  # add prediciton as a key to the message
                    self.q.put(new_msg)  # put new message in the queue

                def extend(self, items):
                    # append loop
                    for item in items:
                        self.append(item)
            
            q_wrapped = QWrap(q_out)

            try:
                # do predictions
                self.mgr.model.predict_video(video_path,
                                             batch_size=self.mgr.batch_size.get(msg),
                                             pred_every=self.mgr.pred_every.get(msg),
                                             show_pbar=False,
                                             global_aggregator=q_wrapped)
            except:
                self.logger.exception(f'An error occurred during detection on {video_path}')
            else:
                self.logger.info(f'detection completed: {video_path}')

            # done with task, empty current task indicator
            cur_detection_task = None

            # if quitting, don't get next item.
            if self._exit_flag:
                break
        
        q_out.put(self.pipeline.STOP_MSG)  # tell tracker dispatcher that no more predicitons are coming


class TrackerStep(ExpertPipelineStep):
    def run(self, q_in: Queue, q_out: AggQueue):
        q_dict = self.mgr.tracker_dispatch_dict

        self.logger.info('trackers thread ready for input')

        # global exit indicator flag
        global exit_all

        # for each new movie launch a thread from the thread pool. queue iteration must be in thread pool
        # executor context.
        with ThreadPoolExecutor(thread_name_prefix='Tracker') as executor:

            # iterate queue messages until the STOP message
            for msg in iter(q_in.get, self.pipeline.STOP_MSG):
                
                # get video path (used as video identifier)
                video_path = msg[VIDEO_PATH_KEY]

                # handle new video
                if video_path not in q_dict:
                    # add video to queue
                    q_dict[video_path] = Queue()

                    # set video annotations output path
                    output_dir = self.mgr.output_dir.get(msg)
                    output_path = None if not output_dir else os.path.join(
                        output_dir,
                        os.path.splitext(os.path.basename(video_path))[0] + "__anno_detect_n_track"
                    )

                    # execute tracking in stand-alone thread.
                    executor.submit(
                        self.single_tracker,
                        q_dict[video_path],
                        q_in,
                        video_path,
                        self.mgr.pred_every.get(msg),
                        self.mgr.merge_iou_thresh.get(msg),
                        self.mgr.tracker_type.get(msg),
                        self.mgr.refresh_on_detect.get(msg),
                        self.mgr.output_style.get(msg),
                        output_path,
                        msg[FPS_KEY]
                    )

                # get prediction
                pred = msg[PREDICTION_KEY]

                # handle "end of tracking" messages from the tracker thread
                if pred == TRACK_COMPLETE_MSG or pred == TRACK_FAILED_MSG or pred == TRACK_STOPPED_MSG:
                    self.logger.info(f'tracking {"completed" if pred == TRACK_COMPLETE_MSG else "failed" if pred == TRACK_FAILED_MSG else "stopped"}: {video_path}')
                    q_dict.pop(video_path)
                else:
                    if pred == self.pipeline.STOP_MSG:  # external stop message added. will stop tracking when this message is found in the tracker's queue
                        self.logger.info(f'got STOP command for tracking video: {video_path}')
                    else:  # got prediction frames
                        self.logger.info(f'got detections frame for video: {video_path}')
                    
                    # put predicitons in the correct queue
                    q_dict[video_path].put(pred)

                # if quitting, don't get next item.
                if self._exit_flag:
                    break
            
            # tell all tracker threads to stop when done with the current frames.
            self.logger.info('sending STOP command to all tracker threads and waiting for them to quit')
            for q in q_dict.values():
                q.put(self.pipeline.STOP_MSG)

    def single_tracker(self, detection_queue, completion_queue, video_path, detect_every,
                       merge_iou_threshold, tracker_type, refresh_on_detect,
                       output_style, output_path, video_fps):
        """
        runner for single thread detection.
        @param; detection_queue: queue to get detection model detections for the given video.
        @param: completion_queue: queue for signalling completion of tracking of the given video.
        @param: video_path: path to the video on which to track objects.
        @param: detect_every: how many frames to track before accepting detection model detections.
        @param: merge_iou_threshold: the IOU score threhold for merging items during tracking.
        @param: refresh_on_detect: if True, removes all tracked items that were not found by the detection model.
        @param: output_style: the method of saving the output.
        @param: output_path: the path in which to save the output annotations.
        """
        self.logger.info(f'start tracking: {video_path}')
        try:
            track_data = at.tracking_utils.MultiTracker.track_video_objects(
                video_path=video_path,
                detection_model=detection_queue,
                detect_every=detect_every,
                merge_iou_threshold=merge_iou_threshold,
                tracker_type=tracker_type,
                refresh_on_detect=refresh_on_detect,
                show_pbar=False,
                logger=self.logger
            )
        except at.tracking_utils.EarlyStopException:
            self.logger.info(f'Tracking stopped early for: {video_path}')
            completion_queue.put({VIDEO_PATH_KEY: video_path, PREDICTION_KEY: TRACK_STOPPED_MSG})
        except:
            self.logger.exception(f'An error occurred during tracking on {video_path}')
            completion_queue.put({VIDEO_PATH_KEY: video_path, PREDICTION_KEY: TRACK_FAILED_MSG})
        else:
            try:
                self.__save_predictions(video_path, video_fps, track_data, output_style, output_path)
            except:
                self.logger.exception(f'An error occurred while saving tracking data for {video_path}')
                completion_queue.put({VIDEO_PATH_KEY: video_path, PREDICTION_KEY: TRACK_FAILED_MSG})
            else:
                completion_queue.put({VIDEO_PATH_KEY: video_path, PREDICTION_KEY: TRACK_COMPLETE_MSG})

    def __save_predictions(self, video_path, fps, preds, output_style, output_path):
        """
        Save tracking output in the desired format
        @param: data: the tracking data (dictionary) to save
        @param: output_style: the saving method:
                    "json" - save as JSON to given `output_path`.
                    "anno" - save as annotated video (mp4)
                    "arango" - save as entry in arango DB.
        @param: output_path: local save path in case of "FILE" `output_style`.
        """
        if OUTPUT_STYLE_JSON in output_style:  # save as JSON
            with open(output_path + '.json', 'w') as f:
                json.dump(preds, f, indent=4)
            self.logger.info(f'successfully saved json annotation for {video_path}')

        if OUTPUT_STYLE_ANNO in output_style:  # save as annotated video
            TrackerAnnotator().annotate_video(video_path, preds, output_path, video_fps=fps, show_pbar=False)
            self.logger.info(f'successfully saved video annotation for {video_path}')

        if OUTPUT_STYLE_ARANGO in output_style:  # save as DB entry
            nodes_saved = self.api.save_track_data_to_scenegraph(video_path, preds)
            self.logger.info(f'successfully saved {nodes_saved} DB entries for {video_path}')
            
        
class ArangoStep(ExpertPipelineStep):
    def run(self, q_in: Queue, q_out: AggQueue):
        self.logger.info('Arango client ready')

        self.mgr.logger.info('ready to receive remote commands')
        for movie_id in self.api.scheduler_loop():
            self.mgr.logger.info(f'got movie from scheduler: {movie_id}')
            q_out.put(self.mgr._get_video_msg(movie_id, is_remote=True))

