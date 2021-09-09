"""
Strictly following the tutorial:
https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb
"""

from ..detector import PretrainedVideoPredictor

import numpy as np
import os
import tensorflow as tf
import pathlib

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# ================== Datasets label maps ==================

# all pre-trained datasets label maps can be found here
LABEL_MAPS_DIR = os.path.join(os.path.dirname(__file__), 'models', 'research', 'object_detection', 'data')

LABEL_MAP_COCO = 'mscoco_label_map.pbtxt'
LABEL_MAP_OID_V4 = 'oid_v4_label_map.pbtxt'


# ================== Tensorflow pretrained models ==================

# COCO detection
CFG_COCO_DETECTION_SSD_MOBILENET_V1 = 'ssd_mobilenet_v1_coco_2017_11_17'

# OID V4 detection
CFG_OID_V4_DETECTION_SSD_MOBILENET_V2 = 'ssd_mobilenet_v2_oid_v4_2018_12_12'  # fast and dirty
CFG_OID_V4_DETECTION_FerRCNN_INCEPTION_V2 = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12'  # slow and accurate


MODEL_TO_DATASET_MAP = {
    CFG_COCO_DETECTION_SSD_MOBILENET_V1: LABEL_MAP_COCO,
    CFG_OID_V4_DETECTION_SSD_MOBILENET_V2: LABEL_MAP_OID_V4,
    CFG_OID_V4_DETECTION_FerRCNN_INCEPTION_V2: LABEL_MAP_OID_V4
}


class VideoPredictor(PretrainedVideoPredictor):
    """
    A video wrapper for the tensorflow object detection API.
    """
    def __init__(self,
                 pretrained_model_cfg=CFG_OID_V4_DETECTION_FerRCNN_INCEPTION_V2,
                 confidence_threshold=0.5):
        super().__init__(pretrained_model_cfg, confidence_threshold)

    def load_model(self):
        """
        load a model with pretrained weights and accompanying label map.
        @return: a tuple (model, label_map)
        """
        # load label map
        path_to_label_map = os.path.join(LABEL_MAPS_DIR, MODEL_TO_DATASET_MAP[self.model_cfg])
        label_map = label_map_util.create_category_index_from_labelmap(path_to_label_map,
                                                                use_display_name=True)

        # get remote model dir (if necessary)
        base_url = 'http://download.tensorflow.org/models/object_detection/'
        model_file = self.model_cfg + '.tar.gz'
        model_dir = tf.keras.utils.get_file(
            fname=self.model_cfg, 
            origin=base_url + model_file,
            untar=True)

        # load pretrained model
        model_dir = pathlib.Path(model_dir)/"saved_model"
        model = tf.saved_model.load(str(model_dir))

        # perform one null prediciton in order to initialize the model
        print('initializing model')
        self.model = model  # must have model+label_map attributes for prediciton
        self.label_map = label_map
        self.predict_single_frame(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))

        return model, label_map

    def predict_batch(self, image_batch: np.ndarray):
        """
        Perform prediciton on a batch of images.
        @param: image_batch: a numpy array of shape (batch_size, H, W, C)
        """
        assert image_batch.ndim == 4, 'must be batch of RGB images'
        batch_size, height, width = image_batch.shape[:3]

        input_tensor = tf.convert_to_tensor(image_batch)

        # Run inference
        model_fn = self.model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = output_dict.pop('num_detections')

        output_dict = {key: [value[i, :int(num_detections[i])].numpy()
                             for i in range(batch_size)]
                        for key,value in output_dict.items()}
        
        # output_dict = {key: value.numpy()
        #                for key,value in output_dict.items()}

        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = [output_dict['detection_classes'][i].astype(np.int64)
                                            for i in range(batch_size)]

        # convert to list of dicts
        preds = [{key: value[i] for key, value in output_dict.items()} for i in range(batch_size)]

        # keep only those above the confidence threhsold
        if self.confidence_t:
            for p in preds:
                keep_msk = p['detection_scores'] > self.confidence_t
                p['detection_scores'] = p['detection_scores'][keep_msk]
                p['detection_classes'] = p['detection_classes'][keep_msk]
                p['detection_boxes'] = p['detection_boxes'][keep_msk]

        # set to uniform API
        preds = [self.__unify(p, width, height) for p in preds]
            
        return preds

    def __unify(self, pred, w, h):
        """
        update a prediction output dictionary to conform to the uniform API:
        {
            "detection_boxes": (x, y, w, h),
            "detection_classes": the string name of the predicted class,
            "detection_scores": prediction confidence probability in [0, 1]
        }
        @param: pred: a tensorflow object detection API prediction output dictionary.
        @param: w: the width of the predicted image.
        @param: h: the height of the predicted image.
        @return: the prediction in uniform API format.
        """
        pred.pop('num_detections')  # irrelevant field
                
        # original box values between 0 and 1 in (ymin, xmin, ymax, xmax) format
        (pred['detection_boxes'][:, ::2],
         pred['detection_boxes'][:, 1::2]) = (pred['detection_boxes'][:, 1::2] * w,  # times height
                                              pred['detection_boxes'][:, ::2] * h)   # times width

        # change box format xyxy --> xywh
        pred['detection_boxes'][:, 2:] = pred['detection_boxes'][:, 2:] - pred['detection_boxes'][:, :2]
        
        # map classes to strings
        pred['detection_classes'] = np.array([
            self.label_map[c]['name'] for c in pred['detection_classes']
        ])

        return pred


#TODO: use tensorRT to optimize models
def optimize_model(model_dir):
    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<32))
    conversion_params = conversion_params._replace(precision_mode="FP16")
    conversion_params = conversion_params._replace(maximum_cached_engines=100)
    
    print(model_dir)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=model_dir,
        conversion_params=conversion_params)
    converter.convert()
    
    def my_input_fn():
        # Input for a single inference call, for a network that has two input tensors:
        inp1 = np.random.normal(size=(8, 16, 16, 3)).astype(np.float32)
        inp2 = np.random.normal(size=(8, 16, 16, 3)).astype(np.float32)
        yield (inp1, inp2)
    
    new_model_dir = os.path.join('.', os.path.basename(model_dir))

    print(new_model_dir)
    converter.build(input_fn=my_input_fn)
    converter.save(new_model_dir)
    print(os.listdir(new_model_dir))

    return new_model_dir
