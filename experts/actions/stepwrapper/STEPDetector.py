import os
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from tqdm import tqdm
from .customized_datasets import CustomizedFrameImagesFolder, CustomizedVideoFile
from .STEP.data.customize import detection_collate
from .STEP.data.augmentations import BaseTransform
from .STEP.external.maskrcnn_benchmark.roi_layers import nms
from .STEP.models import BaseNet, ROINet, TwoBranchNet, ContextNet
from .STEP.utils.tube_utils import valid_tubes, compute_box_iou
from .STEP.utils.utils import inference

STEP_PRETRAINED_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'STEP', 'pretrained',
                                          'ava_step.pth')


class STEPDetector:
    """
    A video wrapper for the tensorflow object detection API.
    """

    def __init__(self, checkpoint_path: str = STEP_PRETRAINED_MODEL_PATH):
        """
        Create a predictor of the given pretrained model config.
        @param: pretrained_model_cfg: the name of a model config file. use CFG_ constants.
        @param: confidence_threshold: only save predictions with a confidence score above this threshold.
        """
        print('loading model')
        self.checkpoint_path = checkpoint_path
        self.model, self.label_map = self.load_model()

    def load_model(self):
        """
        load a model with pretrained weights and accompanying label map.
        @return: a tuple (model, label_map)
        """
        if os.path.isfile(self.checkpoint_path):
            print ("Loading pretrained model from %s" % self.checkpoint_path)
            map_location = 'cuda:0'
            checkpoint = torch.load(self.checkpoint_path, map_location=map_location)
            args = checkpoint['cfg']
        else:
            raise ValueError("Pretrain model not found!", self.checkpoint_path)

        gpu_count = torch.cuda.device_count()
        nets = OrderedDict()
        # backbone network
        nets['base_net'] = BaseNet(args)
        # ROI pooling
        nets['roi_net'] = ROINet(args.pool_mode, args.pool_size)

        # detection network
        for i in range(args.max_iter):
            if args.det_net == "two_branch":
                nets['det_net%d' % i] = TwoBranchNet(args)
            else:
                raise NotImplementedError
        if not args.no_context:
            # context branch
            nets['context_net'] = ContextNet(args)

        for key in nets:
            nets[key] = nets[key].cuda()

        nets['base_net'] = torch.nn.DataParallel(nets['base_net'])
        if not args.no_context:
            nets['context_net'] = torch.nn.DataParallel(nets['context_net'])
        for i in range(args.max_iter):
            nets['det_net%d' % i].to('cuda:%d' % ((i+1)%gpu_count))
            nets['det_net%d' % i].set_device('cuda:%d' % ((i+1)%gpu_count))

        # load pretrained model 
        nets['base_net'].load_state_dict(checkpoint['base_net'])
        if not args.no_context and 'context_net' in checkpoint:
            nets['context_net'].load_state_dict(checkpoint['context_net'])
        for i in range(args.max_iter):
            pretrained_dict = checkpoint['det_net%d' % i]
            nets['det_net%d' % i].load_state_dict(pretrained_dict)

        for _, net in nets.items():
            net.eval()

        def model_forward(batch):
            images, tubes = batch
            images = images.cuda()

            # get conv features
            conv_feat = nets['base_net'](images)
            context_feat = None
            if not args.no_context:
                context_feat = nets['context_net'](conv_feat)

            # do inference
            history, _ = inference(args,
                                   conv_feat,
                                   context_feat,
                                   nets,
                                   args.max_iter,
                                   tubes)
            return history
        label_map = {label_idx: args.id2class[label_id] for label_idx, label_id in args.label_dict.items()}

        self._args = args
        self._nets = nets
        return model_forward, label_map

    def predict_video(self,
                      path_to_video,
                      im_format: str = 'frame%04d.jpg',
                      source_fps: int = 25,
                      confidence_threshold=0.4,
                      global_iou_threshold=0.8,
                      show_pbar: bool = True,
                      num_workers=0,
                      global_aggregator=None):

        ################ DataLoader setup #################
        if os.path.isdir(path_to_video):
            dataset = CustomizedFrameImagesFolder(path_to_video, self._args.T,
                                                  self._args.NUM_CHUNKS[self._args.max_iter], source_fps,
                                                  self._args.fps,
                                                  BaseTransform(self._args.image_size, self._args.means,
                                                                self._args.stds, self._args.scale_norm),
                                                  anchor_mode=self._args.anchor_mode, im_format=im_format)
        else:
            dataset = CustomizedVideoFile(path_to_video, self._args.T,
                                          self._args.NUM_CHUNKS[self._args.max_iter], source_fps,
                                          self._args.fps,
                                          BaseTransform(self._args.image_size, self._args.means,
                                                        self._args.stds, self._args.scale_norm),
                                          anchor_mode=self._args.anchor_mode, im_format=im_format)

        dataloader = torch.utils.data.DataLoader(dataset, self._args.batch_size, num_workers=num_workers,
                                                 shuffle=False, collate_fn=detection_collate, pin_memory=True)
        ################ Inference #################

        # save predictions in global aggregator if provided
        if global_aggregator is not None:
            preds = global_aggregator
        else:
            preds = []

        with torch.no_grad():
            for images, tubes, infos in tqdm(dataloader, disable=not show_pbar):
                batch_preds = {}
                _, _, channels, height, width = images.size()
                history = self.model((images, tubes))

                # collect result of the last step
                pred_prob = history[-1]['pred_prob'].cpu()
                pred_prob = pred_prob[:,int(pred_prob.shape[1]/2)]
                pred_tubes = history[-1]['pred_loc'].cpu()
                pred_tubes = pred_tubes[:,int(pred_tubes.shape[1]/2)]
                tubes_nums = history[-1]['tubes_nums']

                # loop for each batch
                tubes_count = 0
                for b in range(len(tubes_nums)):
                    info = infos[b]
                    seq_start = tubes_count
                    tubes_count = tubes_count + tubes_nums[b]

                    cur_pred_prob = pred_prob[seq_start:seq_start+tubes_nums[b]]
                    cur_pred_tubes = pred_tubes[seq_start:seq_start+tubes_nums[b]]

                    # do NMS first
                    all_scores = []
                    all_boxes = []
                    all_idx = []
                    batch_preds[info['fid']] = {
                        "detection_boxes": [],
                        "detection_classes": [],
                        "detection_scores": []
                    }
                    for cl_ind in range(self._args.num_classes):
                        scores = cur_pred_prob[:, cl_ind].squeeze()
                        c_mask = scores.gt(confidence_threshold) # greater than a threshold
                        scores = scores[c_mask]
                        idx = np.where(c_mask.numpy())[0]
                        if len(scores) == 0:
                            all_scores.append([])
                            all_boxes.append([])
                            continue
                        boxes = cur_pred_tubes.clone()
                        l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                        boxes = boxes[l_mask].view(-1, 4)

                        boxes = valid_tubes(boxes.view(-1,1,4)).view(-1,4)
                        keep = nms(boxes, scores, self._args.nms_thresh)
                        boxes = boxes[keep].numpy()
                        scores = scores[keep].numpy()
                        idx = idx[keep]

                        boxes[:, ::2] /= width
                        boxes[:, 1::2] /= height
                        all_scores.append(scores)
                        all_boxes.append(boxes)
                        all_idx.append(idx)

                    # get the top scores
                    scores_list = [(s,cl_ind,j) for cl_ind,scores in enumerate(all_scores) for j,s in enumerate(scores)]
                    if self._args.evaluate_topk > 0:
                        scores_list.sort(key=lambda x: x[0])
                        scores_list = scores_list[::-1]
                        scores_list = scores_list[:self._args.topk]
                    
                    # merge high overlapping boxes (a simple greedy method)
                    merged_result = {}
                    flag = [1 for _ in range(len(scores_list))]
                    for i in range(len(scores_list)):
                        if flag[i]:
                            s, cl_ind, j = scores_list[i]
                            box = all_boxes[cl_ind][j]
                            temp = ([box], [self._args.label_dict[cl_ind]], [s])

                            # find all high IoU boxes
                            for ii in range(i+1, len(scores_list)):
                                if flag[ii]:
                                    s2, cl_ind2, j2 = scores_list[ii]
                                    box2 = all_boxes[cl_ind2][j2]
                                    if compute_box_iou(box, box2) > global_iou_threshold:
                                        flag[ii] = 0
                                        temp[0].append(box2)
                                        temp[1].append(self._args.label_dict[cl_ind2])
                                        temp[2].append(s2)
                            
                            merged_box = np.mean(np.concatenate(temp[0], axis=0).reshape(-1,4), axis=0)
                            key = ','.join(merged_box.astype(str).tolist())
                            merged_result[key] = [(l, s) for l,s in zip(temp[1], temp[2])]

                    for key, labels_and_scores in merged_result.items():
                        labels, scores = zip(*labels_and_scores)
                        box = np.asarray(key.split(','), dtype=np.float32)
                        batch_preds[info['fid']]['detection_boxes'].append(box.tolist())
                        batch_preds[info['fid']]['detection_classes'].append([self._args.id2class[l] for l in labels])
                        batch_preds[info['fid']]['detection_scores'].append([s.item() for s in scores])
                
                batch_preds = [batch_preds[fnum] for fnum in sorted(batch_preds.keys())]
                preds.extend(batch_preds)
        
        return preds