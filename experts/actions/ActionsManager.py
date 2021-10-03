import json
import os
import sys

from queue import Queue
from argparse import ArgumentParser
from logging import Logger

from .ActionAnnotator import ActionAnnotator
from .ActionsAPIUtility import ActionsAPIUtility
from .stepwrapper import STEPDetector

# import from common
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.ExpertManager import (ExpertManager, ExpertPipeline, ExpertPipelineStep, CLI_command, global_config,
                                  AggQueue, OUTPUT_STYLE_ANNO, OUTPUT_STYLE_ARANGO, OUTPUT_STYLE_JSON)
from common.video_util import VideoInfo

# 
ACTION_CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD_FOR_DUPLICATES = 0.8
DEFAULT_FPS_FOR_FRAME_FOLDER = 25


# task message keys
VIDEO_PATH_KEY = 'video_path'
IS_REMOTE_KEY = 'is_remote'
FPS_KEY = 'fps'
NUM_FRAMES = 'num_frames'

# step names
ACTION_DETECTOR = 'Actions Detector'
SCHEDULER = 'Scheduler'

class ActionsManager(ExpertManager):

    def initialize(self):
        self.api = ActionsAPIUtility()
        self.cur_task = None
        self.step_model = STEPDetector()

    def get_pipeline(self) -> ExpertPipeline:
        detector_step = DetectorStep(ACTION_DETECTOR)
        if self.args.arango:
            arango_step = ArangoStep('Scheduler', is_daemon=True)
            return ExpertPipeline([(arango_step, detector_step)])
        else:
            return ExpertPipeline([(detector_step, detector_step)])

    @CLI_command
    def local(self, line):
        """run action detection on local movie file or frames directory, e.g.: local /movies/scenecliptest00581.avi"""
        if os.path.exists(line):
            detector_q = self.pipeline.incoming_queues[ACTION_DETECTOR]
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
            detector_q = self.pipeline.incoming_queues[ACTION_DETECTOR]
            detector_q.put(self._get_video_msg(line, is_remote=True))
        except ValueError:
            self.logger.exception(f'remote movie does not exist')

    @CLI_command
    def tasks(self, line=''):
        """view enqueued tasks"""
        detector_q = self.pipeline.incoming_queues[ACTION_DETECTOR]
        self.print_n_log(f'tasks queue: {list(detector_q.queue)}')
        self.print_n_log(f'cur task: {self.cur_task}')


    def _get_video_msg(self, video_path, is_remote):
        msg = {VIDEO_PATH_KEY: video_path, IS_REMOTE_KEY: is_remote}
        msg.update(self.get_current_config())

        video_info = VideoInfo(video_path)
        msg[FPS_KEY] = video_info.fps if video_info.fps else self.default_fps.get()
        msg[NUM_FRAMES] = video_info.num_frames

        return msg

    class global_iou_threshold(global_config):
        def __init__(self, default_value=IOU_THRESHOLD_FOR_DUPLICATES):
            super().__init__(default_value)

        def set(self, new_value: str):
            new_value_float = float(new_value)
            assert 0 <= new_value_float <= 1
            self._value = new_value_float

    class confidence_threshold(global_iou_threshold):
        def __init__(self, default_value=ACTION_CONFIDENCE_THRESHOLD):
            super().__init__(default_value=default_value)

    class default_fps(global_config):
        def __init__(self, default_value=25):
            super().__init__(default_value)

        def set(self, new_value: str):
            new_value_int = int(new_value)
            assert new_value_int > 0
            self._value = new_value_int


class DetectorStep(ExpertPipelineStep):
    def run(self, q_in: Queue, q_out: AggQueue):
        self.mgr.print_n_log('detection thread ready for input')

        # iterate until STOP message
        for msg in iter(q_in.get, self.pipeline.STOP_MSG):
            # set current video path
            video_path = msg[VIDEO_PATH_KEY]
            self.cur_task = video_path

             # download if necessary
            if msg[IS_REMOTE_KEY]:
                # download frames or continue if it does not exist
                self.logger.info(f'downloading remote movie: {video_path}')
                num_frames = self.mgr.api.downloadDirectoryFroms3(video_path)
                if num_frames == 0:
                    self.logger.error(f'no frames found under the name {video_path}')
                    self.cur_task = None  # if no frames are found, empty current task indicator
                    continue

            self.logger.info(f'starting action detection: {video_path}')

            logger = self.logger
            class LoggingList:
                def __init__(self):
                    self.lst = []
                    self.count = 0

                def append(self, item):
                    logger.info(f'{video_path}: got actions for frame {self.count + 1}/{msg[NUM_FRAMES]}')
                    self.count += 1
                    self.lst.append(item)

                def extend(self, items):
                    for item in items:
                        self.append(item)

                def __len__(self):
                    return len(self.lst)
                
                def __iter__(self):
                    return iter(self.lst)

            try:
                # do predictions
                preds = self.mgr.step_model.predict_video(video_path,
                                                          source_fps=msg[FPS_KEY],
                                                          confidence_threshold=self.mgr.confidence_threshold.get(msg),
                                                          global_iou_threshold=self.mgr.global_iou_threshold.get(msg),
                                                          show_pbar=False,
                                                          global_aggregator=LoggingList())
            except:
                self.logger.exception(f'An error occurred during action detection on {video_path}')
                preds = None
            else:
                self.logger.info(f'action detection completed: {video_path}')

            if preds is not None:
                preds = {str(i): p for i, p in enumerate(preds)}
                output_path = None if not self.mgr.output_dir.get(msg) else os.path.join(
                    self.mgr.output_dir.get(msg),
                    os.path.splitext(os.path.basename(video_path))[0] + "__anno_action_detection"
                )
                self.__save_predictions(msg, preds, self.mgr.output_style.get(msg), output_path)

            self.cur_task = None

            if self._exit_flag:
                break

    def __save_predictions(self, msg, preds, output_style, output_path):
        """
        Save tracking output in the desired format
        @param: data: the tracking data (dictionary) to save
        @param: output_style: the saving method:
                    "json" - save as JSON to given `output_path`.
                    "anno" - save as annotated video (mp4)
                    "arango" - save as entry in arango DB.
        @param: output_path: local save path in case of "FILE" `output_style`.
        """
        video_path = msg[VIDEO_PATH_KEY]
        fps = msg[FPS_KEY]

        if OUTPUT_STYLE_JSON in output_style:  # save as JSON
            with open(output_path + '.json', 'w') as f:
                json.dump(preds, f, indent=4)
            self.logger.info(f'successfully saved json annotation for {video_path}')

        if OUTPUT_STYLE_ANNO in output_style:  # save as annotated video
            ActionAnnotator().annotate_video(video_path, preds, output_path, video_fps=fps, show_pbar=False)
            self.logger.info(f'successfully saved video annotation for {video_path}')

        if OUTPUT_STYLE_ARANGO in output_style:  # save as DB entry
            nodes_saved = self.api.save_action_data_to_scene_graph(video_path, preds)
            self.logger.info(f'successfully saved {nodes_saved} DB entries for {video_path}')
            
        
class ArangoStep(ExpertPipelineStep):
    def run(self, q_in: Queue, q_out: AggQueue):
        self.logger.info('Arango client ready')

        self.mgr.logger.info('ready to receive remote commands')
        for movie_id in self.api.scheduler_loop():
            self.mgr.logger.info(f'got movie from scheduler: {movie_id}')
            q_out.put(self.mgr._get_video_msg(movie_id, is_remote=True))

