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

import autotracker as at
from remote_utils import RemoteUtility
from my_annotator import annotate_video


# ===== Input Constants =====
# default model configurations
CFG_DEFAULT = {
    at.BACKEND_TFLOW: 'CFG_OID_V4_DETECTION_FerRCNN_INCEPTION_V2',
    at.BACKEND_DETECTRON: 'CFG_COCO_DETECTION_FerRCNN_X101_32x8d_FPN_LR3x'
}

# output save style
OUTPUT_STYLE_JSON = 'json'
OUTPUT_STYLE_ANNO = 'anno'
OUTPUT_STYLE_ARANGO = 'arango'

# default output file
OUTPUT_DEFAULT = './annotations'


# ===== Command Constants =====
# special queue messages for event control
STOP_MSG = 'STOP'
TRACK_COMPLETE_MSG = 'TRACK COMPLETE'
TRACK_STOPPED_MSG = 'TRACK STOPPED'
TRACK_FAILED_MSG = 'TRACK FAILED'

# task message keys
VIDEO_PATH_KEY = 'video_path'
IS_REMOTE_KEY = 'is_remote'
PREDICTION_KEY = 'predictions'
PRED_EVERY_KEY = 'pred_every'
DETECTION_BATCH_SIZE_KEY = 'batch_size'
TRACK_REFRESH_ON_DETECT_KEY = 'refresh_on_detect'
TRACK_MERGE_IOU_THRESH_KEY = 'merge_iou_thresh'
TRACK_TRACKER_TYPE_KEY = 'tracker_type'

# cmd-line cli commands
EXIT_CMD = 'exit'
WAIT_CMD = 'wait'
STATUS_CMD = 'status'
VIEW_DETECT_CMD = 'qdetect'
VIEW_DETECT_PROGRESS_CMD = 'qdetectprog'
VIEW_TRACK_CMD = 'qtrack'
VIEW_TRACK_PROGRESS_CMD = 'qtrackprog'
VIEW_CFG_CMD = 'cfg'
REMOTE_MOVIE_CMD_PREFIX = 'remote:'
SET_CFG_CMD_PREFIX = 'set:'


# ===== Globals =====
# a global switch indicating shutdown
exit_all = False

# a global cotainer for the current video under detection
cur_detection_task = None

# set global logger
logger = logging.getLogger('Engine Manager')

# set global remote comm
remote = None

# general detection and tracker global defaults
pred_every_cfg = 10
detection_batch_size_cfg = 8
track_refresh_on_detect_cfg = False
track_merge_iou_thresh_cfg = 0.5
track_tracker_type_cfg = at.tracking_utils.TRACKER_TYPE_KCF


# ===== Public Functions =====

def run_detector(q_in: Queue, q_out: Queue, model):
    """
    Detection model thread code.
    @param: q_in: input queue for paths to videos on which to detect.
    @param: q_out: queue in which to place predictions
    @param: model: the detection model. must implement `predict_video` function
    """
    logger.info('detection thread ready for input')

    global exit_all
    global cur_detection_task
    for msg in iter(q_in.get, STOP_MSG):
        video_path = msg[VIDEO_PATH_KEY]
        cur_detection_task = video_path

        if video_path.startswith(REMOTE_MOVIE_CMD_PREFIX):
            video_path = video_path[len(REMOTE_MOVIE_CMD_PREFIX):].strip()
            
            # download frames or continue if it does not exist
            logger.info(f'downloading remote movie: {video_path}')
            num_frames = __get_remote().download_arango_by_id(video_path)
            if num_frames == 0:
                logger.error(f'no frames found under the name {video_path}')
                continue

        logger.info(f'starting detection: {video_path}')

        # wrap queues to have the extend and append functions
        class QWrap:
            def __init__(self, q):
                self.q = q
            def append(self, item):
                new_msg = msg.copy()
                new_msg[PREDICTION_KEY] = item
                self.q.put(new_msg)
            def extend(self, items):
                for item in items:
                    self.append(item)

        try:
            model.predict_video(video_path,
                                batch_size=msg[DETECTION_BATCH_SIZE_KEY],
                                pred_every=msg[PRED_EVERY_KEY],
                                show_pbar=False,
                                global_aggregator=QWrap(q_out))
        except:
            logger.exception(f'An error occurred during detection on {video_path}')
        else:
            logger.info(f'detection completed: {video_path}')

        cur_detection_task = None

        if exit_all:
            break
    
    q_out.put(STOP_MSG)  # tell tracker dispatcher that no more predicitons are coming


def run_trackers(q_in, q_dict, output_style, output_dir):
    """
    Tracker dispatcher code.
    @param: q_in: queue for input detections from a detection model
    @param: q_dict: a dictionary of queues for trackers of different clips.
    @param: output_dir: the directory in which to save all annotation outputs.
    """
    # create output dir if necessary
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info('trackers thread ready for input')

    # for each new movie launch a thread from the thread pool
    global exit_all
    with ThreadPoolExecutor(thread_name_prefix='Tracker') as executor:
        for msg in iter(q_in.get, STOP_MSG):
            video_path = msg[VIDEO_PATH_KEY]
            if video_path not in q_dict:  # handle new video
                q_dict[video_path] = Queue()
                output_path = None if not output_dir else os.path.join(
                    output_dir,
                    os.path.splitext(os.path.basename(video_path))[0] + "__anno"
                )
                executor.submit(
                    single_tracker,
                    q_dict[video_path],
                    q_in,
                    video_path,
                    msg[PRED_EVERY_KEY],
                    msg[TRACK_MERGE_IOU_THRESH_KEY],
                    msg[TRACK_TRACKER_TYPE_KEY],
                    msg[TRACK_REFRESH_ON_DETECT_KEY],
                    output_style,
                    output_path
                )

            pred = msg[PREDICTION_KEY]

            # handle "end of tracking" messages from the tracker thread
            if pred == TRACK_COMPLETE_MSG or pred == TRACK_FAILED_MSG or pred == TRACK_STOPPED_MSG:
                logger.info(f'tracking {"completed" if pred == TRACK_COMPLETE_MSG else "failed" if pred == TRACK_FAILED_MSG else "stopped"}: {video_path}')
                q_dict.pop(video_path)
            else:
                if pred == STOP_MSG:  # external stop message added. will stop tracking when this message is found in the tracker's queue
                    logger.info(f'got STOP command for tracking video: {video_path}')
                else:  # got prediction frames
                    logger.info(f'got detections frame for video: {video_path}')
                
                # put predicitons in the correct queue
                q_dict[video_path].put(pred)

            if exit_all:
                break
        
        logger.info('sending STOP command to all tracker threads and waiting for them to quit')
        for q in q_dict.values():
            q.put(STOP_MSG)


def single_tracker(detection_queue, completion_queue, video_path, detect_every,
                   merge_iou_threshold, tracker_type, refresh_on_detect,
                   output_style, output_path):
    """
    runner for single thread detection.
    @param; detection_queue: queue to get detection model detections for the given video.
    @param: completion_queue: queue for signalling completion of tracking of the given video.
    @param: video_path: path to the video on which to track objects.
    @param: output_path: the path in which to save the output annotations.
    """
    logger.info(f'start tracking: {video_path}')
    try:
        track_data = at.tracking_utils.MultiTracker.track_video_objects(
            video_path=video_path,
            detection_model=detection_queue,
            detect_every=detect_every,
            merge_iou_threshold=merge_iou_threshold,
            tracker_type=tracker_type,
            refresh_on_detect=refresh_on_detect,
            show_pbar=False,
            logger=logger
        )
    except at.tracking_utils.EarlyStopException:
        logger.exception(f'Tracking stopped early for: {video_path}')
        completion_queue.put({VIDEO_PATH_KEY: video_path, PREDICTION_KEY: TRACK_STOPPED_MSG})
    except:
        logger.exception(f'An error occurred during tracking on {video_path}')
        completion_queue.put({VIDEO_PATH_KEY: video_path, PREDICTION_KEY: TRACK_FAILED_MSG})
    else:
        try:
            save_output(video_path, track_data, output_style, output_path)
        except:
            logger.exception(f'An error occurred while saving tracking data for {video_path}')
            completion_queue.put({VIDEO_PATH_KEY: video_path, PREDICTION_KEY: TRACK_FAILED_MSG})
        else:
            completion_queue.put({VIDEO_PATH_KEY: video_path, PREDICTION_KEY: TRACK_COMPLETE_MSG})


def save_output(video_path, data, output_style, output_path=None):
    """
    Save tracking output in the desired format
    @param: data: the tracking data (dictionary) to save
    @param: output_style: the saving method:
                "FILE" - save as JSON to given `output_path`.
                "ARANGO" - save as entry in arango DB.
    @param: output_path: local save path in case of "FILE" `output_style`.
    """
    if OUTPUT_STYLE_JSON in output_style:
        with open(output_path + '.json', 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f'successfully saved json annotation for {video_path}')
    if OUTPUT_STYLE_ANNO in output_style:
        annotate_video(video_path, data, output_path, show_pbar=False)
        logger.info(f'successfully saved video annotation for {video_path}')
    if OUTPUT_STYLE_ARANGO in output_style:
        nodes_saved = __get_remote().save_track_data_to_scenegraph(video_path, data)
        logger.info(f'successfully saved {nodes_saved} DB entries for {video_path}')


def main(q_detector_in: Queue, q_detector_out: Queue, q_dict_trackers: dict, output_style: str,
         output_dir: str = None, model_cfg='CFG_OID_V4_DETECTION_FerRCNN_INCEPTION_V2', model_confidence=0.3,
         log_file=None):
    """
    launches all engines.
    @param: q_detector_in: a queue for tasks for the detector.
    @param: q_detector_out: a queue for passing tasks from the detector to the tracker dispatcher.
    @param: q_dict_trackers: a queue for passing tasks to individual tracker threads.
    @param: output_dir: the directory in which to save all annotation outputs.
    @param: model_backend: 'tflow' or 'detectron'.
    @param: model_cfg: the name of the CFG constant in the backend utilities file.
    @param: log_file: where to save logged information. see `setup_logger` for more info.
    @return: a tuple of Thread object (t1, t2) where t1 belongs to the detector and t2 belongs to the trackers
             dispatcher
    """

    __setup_logger(log_file)

    # load model
    logger.info('loading and initializing model')
    logger.info(f'params: backend: {at.active_detection_backend()}, '
                f'cfg: {model_cfg}, confidence: {model_confidence}')
    model = at.detection_utils.VideoPredictor(getattr(at.detection_utils, model_cfg),
                                              confidence_threshold=model_confidence)

    # start detector thread
    t_detector = Thread(target=run_detector, args=(q_detector_in, q_detector_out, model))
    t_detector.name = 'Detector'
    t_detector.start()

    # start tracker dispatch
    t_trackers = Thread(target=run_trackers, args=(q_detector_out, q_dict_trackers, output_style, output_dir))
    t_trackers.name = 'TrackerDispatch'
    t_trackers.start()

    return t_detector, t_trackers


def arango_app(q_detector_in: Queue):
    remote = __get_remote()
    logger.info('Arango client ready')

    logger.info('ready to receive remote commands')
    for movie_id in remote.scheduler_loop():
        logger.info(f'got movie from scheduler: {movie_id}')
        q_detector_in.put(__get_video_msg(f'{REMOTE_MOVIE_CMD_PREFIX} {movie_id}'))


def cmd_line_app(q_detector_in: Queue, q_detector_to_tracker: Queue, q_dict_trackers: dict):
    global exit_all
    global cur_detection_task
    
    # simple keyboard CLI for adding and viewing tasks
    print('infrastructure ready\n\n>>>', end=' ')
    logger.info('CLI ready to receive commands')
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if line:
            logger.info(f'got CLI command: "{line}"')

        if line == WAIT_CMD:  # wait for all tasks to finish then quit
            logger.info('waiting fo all tasks to complete')
            logger.info('Detection queue: ' + str(list(q_detector_in.queue)))
            logger.info('Detection in progress: ' + str(cur_detection_task))
            logger.info('Tracking queue: ' + str(list(q_detector_to_tracker.queue)))
            logger.info('Tracking in progress: ' + str(list(q_dict_trackers.keys())))
            q_detector_in.put(STOP_MSG)
            break

        elif line == EXIT_CMD:  # exit after current tasks finish
            logger.info('waiting for active tasks to complete')
            logger.info('Detection in progress: ' + str(cur_detection_task))
            logger.info('Tracking in progress: ' + str(list(q_dict_trackers.keys())))

            exit_all = True
            q_detector_in.put(STOP_MSG)
            q_detector_to_tracker.put(STOP_MSG)
            for q in q_dict_trackers.values():
                q.queue.clear()
                q.put(STOP_MSG)
            break

        elif line == STATUS_CMD:  # show detection queue
            print('Detection queue:', list(q_detector_in.queue))
            logger.info('Detection queue: ' + str(list(q_detector_in.queue)))
            print('Detection in progress:', cur_detection_task)
            logger.info('Detection in progress: ' + str(cur_detection_task))
            print('Tracking queue:', list(q_detector_to_tracker.queue))
            logger.info('Tracking queue: ' + str(list(q_detector_to_tracker.queue)))
            print('Tracking in progress:', list(q_dict_trackers.keys()))
            logger.info('Tracking in progress: ' + str(list(q_dict_trackers.keys())))
        
        elif line == VIEW_CFG_CMD:  # show current configurations
            print('pred every:', pred_every_cfg)
            logger.info('pred every: ' + str(pred_every_cfg))
            print('batch size:', detection_batch_size_cfg)
            logger.info('batch size: ' + str(detection_batch_size_cfg))
            print('merge iou thresh:', track_merge_iou_thresh_cfg)
            logger.info('merge iou thresh: ' + str(track_merge_iou_thresh_cfg))
            print('tracker type:', track_tracker_type_cfg)
            logger.info('tracker type: ' + str(track_tracker_type_cfg))
            print('refresh on detect:', track_refresh_on_detect_cfg)
            logger.info('refresh on detect: ' + str(track_refresh_on_detect_cfg))
        
        elif line.startswith(SET_CFG_CMD_PREFIX):
            line = line[len(SET_CFG_CMD_PREFIX):].strip()
            try:
                __set_cfg_cmd(line)
            except:
                logger.exception(f'bad cfg set command: "{line}"')
            else:
                logger.info('cfg set')

        elif line == VIEW_DETECT_CMD:  # show detection queue
            print(list(q_detector_in.queue))
            logger.info(str(list(q_detector_in.queue)))

        elif line == VIEW_DETECT_PROGRESS_CMD:  # show detection task in progress
            print(cur_detection_task)
            logger.info(cur_detection_task)

        elif line == VIEW_TRACK_CMD:  # show tracking queue
            print(list(q_detector_to_tracker.queue))
            logger.info(str(list(q_detector_to_tracker.queue)))

        elif line == VIEW_TRACK_PROGRESS_CMD:  # show tracking in progress
            print(list(q_dict_trackers.keys()))
            logger.info(str(list(q_dict_trackers.keys())))

        # send movie file/dir/remote path to detection queue
        elif os.path.exists(line) or line.startswith(REMOTE_MOVIE_CMD_PREFIX):
            q_detector_in.put(__get_video_msg(line))

        elif not line:  # skip empty string
            pass

        else:
            print(f'unsupported command: "{line}"')
            logger.info(f'unsupported command: "{line}"')

        print('>>>', end=' ')
        sys.stdout.flush()


def parse_args():
    """
    parse program arguments
    """
    parser = argparse.ArgumentParser(description='Run ojbect detection and tracking engines')
    parser.add_argument('--backend', '-b',
                        default=None,
                        help='Detection model backend',
                        choices=[at.BACKEND_TFLOW, at.BACKEND_DETECTRON])
    parser.add_argument('--model', '-m',
                        default=None,
                        help='Detection model configuration. Should be '
                             'one of the CFG constants in the chosen model utils.')
    parser.add_argument('--confidence', '-c',
                        default=0.6,
                        type=float,
                        help='path to output log file')
    parser.add_argument('--log', '-l',
                        default=None,
                        help='path to output log file')
    parser.add_argument('--no-arango',
                        action='store_false',
                        dest='arango',
                        default=True,
                        help='use command line client with local movies')
    parser.add_argument('--output-style', '-o',
                        nargs="+",
                        default=None,
                        choices=[OUTPUT_STYLE_JSON, OUTPUT_STYLE_ANNO, OUTPUT_STYLE_ARANGO],
                        help='The method for saving the tracking output')
    parser.add_argument('--output-dir', '-d',
                        default=None,
                        help='For use only when --output-style "json" or "anno" are set. Indicates wher to save '
                             'the annotations.')

    # parse then set defaults
    parsed_args = parser.parse_args()
    parsed_args = __handle_args_defaults(parsed_args)

    print('running with args:')
    for arg in vars(parsed_args):
        print(f'{arg}={getattr(parsed_args, arg)}')
    print()
    
    return parsed_args


#===== Private Functions =====

def __setup_logger(log_file):
    """
    configure the global logger:
    - write DEBUG+ level to given `log_file`
    - write ERROR+ level to stderr
    - format: [time][thread name][log level]: message
    @param log_file: the file to which we wish to write. if an existing dir is given, log to a file
                     labeled with the curent date and time. if None, use the current working directory.
    """
    # set general logging level to debug
    logger.setLevel(logging.DEBUG)

    # choose logging format
    formatter = logging.Formatter('[%(asctime)s][%(threadName)s][%(levelname)s]: %(message)s')

    # create and add file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create stderr stream handler
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.ERROR)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    title = '===== Detector-Tracker Engine ====='
    logger.info('=' * len(title))
    logger.info(title)
    logger.info('=' * len(title))


def __get_remote():
    global remote
    if remote is None:
        remote = RemoteUtility()

    return remote


def __handle_args_defaults(parsed_args):
    # === Detection Defaults ===
    # no backend or config chosen. use default
    if parsed_args.backend is None and parsed_args.model is None:
        parsed_args.backend = at.active_detection_backend()
        parsed_args.model = CFG_DEFAULT[parsed_args.backend]

    # no config chosen. use default config for given backend
    elif parsed_args.model is None:
        at.set_active_backend(parsed_args.backend)
        parsed_args.model = CFG_DEFAULT[parsed_args.backend]
    
    # no backend chosen. find backend that has given config
    elif parsed_args.backend is None:
        possible_backends = at.backends_with_config(parsed_args.model)
        if not possible_backends:
            raise ValueError(f'Model configuration {parsed_args.model} not found in any backend available in this evironment')
        parsed_args.backend = possible_backends[0]
        at.set_active_backend(parsed_args.backend)

    # chosen backend and config. check compatibility
    else:
        at.set_active_backend(parsed_args.backend)
        if not hasattr(at.detection_utils, parsed_args.model):
            raise ValueError(f'Given model backend and config are incopatible in the current environment: {parsed_args.backend} - {parsed_args.model}')

    # === Logging Defaults ===
    # choose default logging location
    if not parsed_args.log:
        parsed_args.log = os.path.join('.', datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.log')
    elif os.path.isdir(args.log):
        parsed_args.log = os.path.join(parsed_args.log, datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.log')
    else:
        os.makedirs(os.path.dirname(parsed_args.log), exist_ok=True)

    # === Output Defaults ===
    # default output to arango without local save
    if not parsed_args.output_style and not parsed_args.output_dir:
        if parsed_args.arango:
            parsed_args.output_style = [OUTPUT_STYLE_ARANGO]
        else:
            parsed_args.output_style =  [OUTPUT_STYLE_JSON]
            parsed_args.output_dir = OUTPUT_DEFAULT

    # got local output save location. save as JSON
    elif not parsed_args.output_style:
        parsed_args.output_style = [OUTPUT_STYLE_JSON]
    
    # got only output style. set output location if necessary
    elif not parsed_args.output_dir:
        if OUTPUT_STYLE_JSON in parsed_args.output_style or OUTPUT_STYLE_ANNO in parsed_args.output_style:
            parsed_args.output_dir = OUTPUT_DEFAULT
    
    # method arango with output file is incompatible
    elif parsed_args.output_style == [OUTPUT_STYLE_ARANGO]:
        print('Warning: output style "arango" does not save locally and will not save output to given path: {parsed_args.output_dir}')
    
    return parsed_args


def __set_cfg_cmd(cfg_line):
    cfg_name, value = cfg_line.split('=')
    if cfg_name == PRED_EVERY_KEY:
        v = int(value)
        assert v > 0
        global pred_every_cfg
        pred_every_cfg = v
    elif cfg_name == DETECTION_BATCH_SIZE_KEY:
        v = int(value)
        assert v > 0
        global detection_batch_size_cfg
        detection_batch_size_cfg = v
    elif cfg_name == TRACK_TRACKER_TYPE_KEY:
        v = getattr(at.tracking_utils, f'TRACKER_TYPE_{value.upper()}')
        global track_tracker_type_cfg
        track_tracker_type_cfg = v
    elif cfg_name == TRACK_MERGE_IOU_THRESH_KEY:
        v = float(value)
        assert 0 < v <= 1
        global track_merge_iou_thresh_cfg
        track_merge_iou_thresh_cfg = v
    elif cfg_name == TRACK_REFRESH_ON_DETECT_KEY:
        v = eval(value)
        assert isinstance(v, bool)
        global track_refresh_on_detect_cfg
        track_refresh_on_detect_cfg =v


def __get_video_msg(video_path):
    if video_path.startswith(REMOTE_MOVIE_CMD_PREFIX):
        is_remote = True
        video_path = video_path[len(REMOTE_MOVIE_CMD_PREFIX):].strip()
    else:
        is_remote = False
    
    return {
        VIDEO_PATH_KEY: video_path,
        IS_REMOTE_KEY: is_remote,
        PRED_EVERY_KEY: pred_every_cfg,
        DETECTION_BATCH_SIZE_KEY: detection_batch_size_cfg,
        TRACK_REFRESH_ON_DETECT_KEY: track_refresh_on_detect_cfg,
        TRACK_MERGE_IOU_THRESH_KEY: track_merge_iou_thresh_cfg,
        TRACK_TRACKER_TYPE_KEY: track_tracker_type_cfg
    }


# ===== Main Code =====

if __name__ == "__main__":
    # parse cmd-line args
    args = parse_args()

    # initialize queues
    detector_q_in, detector_to_tracer_q = Queue(), Queue()
    trackers_q_dict = defaultdict(Queue)

    # start engines and infrastructure
    det_thread, track_thread = main(q_detector_in=detector_q_in,
                                    q_detector_out=detector_to_tracer_q,
                                    q_dict_trackers=trackers_q_dict,
                                    output_style=args.output_style,
                                    output_dir=args.output_dir,
                                    model_cfg=args.model,
                                    model_confidence=args.confidence,
                                    log_file=args.log)

    if args.arango:
        scheduler_thread = Thread(target=arango_app,
                                  args=(detector_q_in,))
        scheduler_thread.name = 'Scheduler'
        scheduler_thread.daemon = True  # set up as daemon so that it will quit when the program terminates.
        scheduler_thread.start()
        print('scheduler is running')

    cmd_line_app(q_detector_in=detector_q_in,
                 q_detector_to_tracker=detector_to_tracer_q,
                 q_dict_trackers=trackers_q_dict)

    # wait for threads to close
    det_thread.join()
    print('detector quit successfully')
    logger.info('detector quit successfully')

    track_thread.join()
    print('tracker quit successfully')
    logger.info('tracker quit successfully')
