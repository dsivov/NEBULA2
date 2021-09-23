import numpy as np
import torch

from ..detector import PretrainedVideoPredictor

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

# ================== Detectron2 pretrained models ==================

# COCO detection
CFG_COCO_DETECTION_FerRCNN_R50_C4_LR1x = 'COCO-Detection/faster_rcnn_R_50_C4_1x.yaml'
CFG_COCO_DETECTION_FerRCNN_R50_DC5_LR1x = 'COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml'
CFG_COCO_DETECTION_FerRCNN_R50_FPN_LR1x = 'COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml'
CFG_COCO_DETECTION_FerRCNN_R50_C4_LR3x = 'COCO-Detection/faster_rcnn_R_50_C4_3x.yaml'
CFG_COCO_DETECTION_FerRCNN_R50_DC5_LR3x = 'COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml'
CFG_COCO_DETECTION_FerRCNN_R50_FPN_LR3x = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
CFG_COCO_DETECTION_FerRCNN_R101_C4_LR3x = 'COCO-Detection/faster_rcnn_R_101_C4_3x.yaml'
CFG_COCO_DETECTION_FerRCNN_R101_DC5_LR3x = 'COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml'
CFG_COCO_DETECTION_FerRCNN_R101_FPN_LR3x = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
CFG_COCO_DETECTION_FerRCNN_X101_32x8d_FPN_LR3x = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'  # newest
CFG_COCO_DETECTION_RETINA_R50_FPN_LR1x = 'COCO-Detection/retinanet_R_50_FPN_1x.yaml'
CFG_COCO_DETECTION_RETINA_R50_FPN_LR3x = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'
CFG_COCO_DETECTION_RETINA_R101_FPN_LR3x = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'
CFG_COCO_DETECTION_RPN_R50_C4_LR1x = 'COCO-Detection/rpn_R_50_C4_1x.yaml'
CFG_COCO_DETECTION_RPN_R50_FPN_LR1x = 'COCO-Detection/rpn_R_50_FPN_1x.yaml'
CFG_COCO_DETECTION_FRCNN_R50_FPN_LR1x = 'COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml'

# COCO panoptic segmentation
CFG_COCO_PANOPTIC_FPN_R50_LR1x = 'COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml'
CFG_COCO_PANOPTIC_FPN_R50_LR3x = 'COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml'
CFG_COCO_PANOPTIC_FPN_R101_LR3x = 'COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'  # newest

# LVIS https://www.lvisdataset.org/
CFG_LVIS_SEGMENTATION_MRCNN_R50_FPN_LR1x = 'LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml'
CFG_LVIS_SEGMENTATION_MRCNN_R101_FPN_LR1x = 'LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml'
CFG_LVIS_SEGMENTATION_MRCNN_X101_32x8d_FPN_LR1x = 'LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml'  # newest

# Citiscapes https://www.cityscapes-dataset.com/
CFG_CITISCAPE_SEGMENTATION_MRCNN_R50_FPN = 'Cityscapes/mask_rcnn_R_50_FPN.yaml'

# Pascal VOC https://paperswithcode.com/dataset/pascal-voc
CFG_PASCAL_DETECTION_FerRCNN_R50_C4 = 'PascalVOC-Detection/faster_rcnn_R_50_C4.yaml'

class VideoPredictor(PretrainedVideoPredictor):
    """
    A video wrapper for the tensorflow object detection API.
    """

    def __init__(self,
                 pretrained_model_cfg = CFG_COCO_DETECTION_FerRCNN_X101_32x8d_FPN_LR3x,
                 confidence_threshold=0.5,
                 device=None):
        """
        Create a predictor of the given pretrained model config.
        @param: pretrained_model_cfg: the name of a model config file. use CFG_ constants.
        @param: confidence_threshold: only save predictions with a confidence score above this threshold.
        @param: device: the inference device (CPU or GPU)
        """
        super().__init__(pretrained_model_cfg, confidence_threshold)
        
        # move model to device
        if not device:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.model.to(device)

        # keep CPU device ready for quick GPU detach.
        self.cpu_device = torch.device("cpu")
    
    def load_model(self):
        # create cfg for model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model_cfg))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_cfg)
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_t
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_t
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.confidence_t

        # load model
        model = DefaultPredictor(cfg).model

        # get label map from dataset metadata
        label_map = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused").thing_classes

        return model, label_map

    def predict_batch(self, image_batch: np.ndarray):
        # convert batch to tensor object. reshape to (batch_size, C, H, W), which is the required format
        # for detectron.
        image_batch_torch = torch.as_tensor(image_batch.astype("float32").transpose(0, 3, 1, 2)).to(self.device)
        height, width = image_batch_torch.shape[2:]  # save height and width for output format

        # change image batch to a list of dictionaries for detectron input format.
        image_batch_formatted = []
        for img in image_batch_torch:
            image_batch_formatted.append({'image': img, 'width': width, 'height': height})

        # perform predictions
        with torch.no_grad():
            preds = self.model(image_batch_formatted)

        # format output for uniform output style
        for i, p in enumerate(preds):
            # immediately move instances to CPU to avoid GPU OOM error.
            new_p = {'instances': p['instances'].to(self.cpu_device)}

            # set to uniform API
            new_p = self.__unify(new_p)

            preds[i] = new_p
        
        return preds

    def __unify(self, pred):
        """
        update a prediction output dictionary to conform to the uniform API:
        {
            "detection_boxes": (x, y, w, h),
            "detection_classes": the string name of the predicted class,
            "detection_scores": prediction confidence probability in [0, 1]
        }
        @param: pred: a detectron2 API prediction output dictionary.
        @return: the prediction in uniform API format.
        """
        # detech relevant values from GPU and save as numpy array
        new_p = {
            'detection_boxes': pred['instances'].pred_boxes.tensor.detach().numpy(),
            'detection_scores': pred['instances'].scores.detach().numpy(),
            'detection_classes': pred['instances'].pred_classes.detach().numpy()
        }

        # convert class number to name
        new_p['detection_classes'] = np.array([
            self.label_map[c] for c in new_p['detection_classes']
        ])

        # change box format xyxy --> xywh
        new_p['detection_boxes'][:, 2:] = new_p['detection_boxes'][:, 2:] - new_p['detection_boxes'][:, :2]

        return new_p
