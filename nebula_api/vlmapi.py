from re import I
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

from pathlib import Path
from PIL import Image
from nebula_api.nebula_enrichment_api import NRE_API
from experts.common.RemoteAPIUtility import RemoteAPIUtility

from nebula_api.mdmmt_api.mdmmt_api import MDMMT_API

from nebula_api.clip_api import CLIP_API
from vcomet.vcomet import VCOMET_KG

class VLM_API:
    def __init__(self):
        self.clip_vit = CLIP_API('vit')
        self.clip_rn = CLIP_API('rn')
        self.vcomet = VCOMET_KG()
        self.remote_api = RemoteAPIUtility()
        self.mdmmt_api = MDMMT_API()
        self.available_class_names = ['clip_vit', 'clip_rn', 'mdmmt_max', 'mdmmt_mean']
        print(f"Available class names: {self.available_class_names}")
    
    def download_and_get_minfo(self, mid, to_print=False):
        # Download the video locally
        fps, url_link = self.vcomet.download_video_file(mid)
        movie_info = self.remote_api.get_movie_info(mid)
        fn = self.vcomet.temp_file
        if to_print:
            print(f"Movie info: {movie_info}")
            print(f"fn path: {fn}")
        return movie_info, fps, fn

    # Prepare arguments to call MDMMT MAX or MDMMT MIN
    def prepare_mdmmt_args(self, movie_info, scene_element, fps, class_name):
        vggish_model, vmz_model, clip_model, model_vid = self.mdmmt_api.vggish_model, \
                                                         self.mdmmt_api.vmz_model, \
                                                         self.mdmmt_api.clip_model, \
                                                         self.mdmmt_api.model_vid
        if scene_element < len(movie_info['scene_elements']):
            scene_element_to_frames = movie_info['scene_elements'][scene_element]
        else:
            raise Exception("Scene element wasn't found, probably the stage is too big, try a lower number.")
        t_start = scene_element_to_frames[0] / fps
        t_end = scene_element_to_frames[1] / fps
        encode_type = 'mean'
        if class_name == 'mdmmt_max':
            encode_type = 'max'
        return vggish_model, vmz_model, clip_model, model_vid, t_start, t_end, fps, encode_type


    def encode_video(self, mid, scene_element, class_name='clip_rn'):
        movie_info, fps, fn = self.download_and_get_minfo(mid, to_print=True)
        path = fn
        if class_name == 'mdmmt_max' or class_name == 'mdmmt_mean':
            vggish_model, vmz_model, clip_model, model_vid, t_start, t_end, fps, encode_type =  \
                self.prepare_mdmmt_args(movie_info, scene_element, fps, class_name)

        if class_name == 'clip_rn':
            vid_embedding = self.clip_rn.clip_encode_video(fn, mid, scene_element)
        elif class_name == 'clip_vit':
            vid_embedding = self.clip_vit.clip_encode_video(fn, mid, scene_element)
        elif class_name == 'mdmmt_max':
            vid_embedding = self.mdmmt_api.encode_video(vggish_model, vmz_model, clip_model, model_vid, path, t_start, t_end, fps, encode_type)
        elif class_name == 'mdmmt_mean':
            vid_embedding = self.mdmmt_api.encode_video(vggish_model, vmz_model, clip_model, model_vid, path, t_start, t_end, fps, encode_type)
        else:
            print(f"Error! Available class names: {self.available_class_names}")
            raise Exception("Class name you entered was not found.")
        return vid_embedding
    
    def encode_text(self, text, class_name='clip_rn'):
        if class_name == 'clip_rn':
            text_embedding = self.clip_rn.clip_batch_encode_text(text)
        elif class_name == 'clip_vit':
            text_embedding = self.clip_vit.clip_batch_encode_text(text)
        elif class_name == 'mdmmt_max':
            text_embedding = self.mdmmt_api.batch_encode_text(text)
        elif class_name == 'mdmmt_mean':
            text_embedding = self.mdmmt_api.batch_encode_text(text)
        else:
            print(f"Error! Available class names: {self.available_class_names}")
            raise Exception("Class name you entered was not found.")
        return text_embedding


    


def main():
    vlm_api = VLM_API()

    text = ['hand',
            'picture of a hand'
        ]

    # Encode video & text of clip_rn
    print("Encoding video and text of CLIP_RN")
    vlm_api.encode_video(mid="Movies/114207205", scene_element=0, class_name='clip_rn')
    text_feat = vlm_api.encode_text(text, class_name='clip_rn')
    print(f"Length of CLIP_RN text embeddings: {len(text_feat)}")
    print("----------------------")

    print("/nEncoding video and text of CLIP_VIT")
    # Encode video & text of clip_vit
    vlm_api.encode_video(mid="Movies/114207205", scene_element=0, class_name='clip_vit')
    text_feat = vlm_api.encode_text(text, class_name='clip_vit')
    print(f"Length of CLIP_VIT text embeddings: {len(text_feat)}")
    print("----------------------")

    print("/nEncoding video and text of MDMMT_MAX")
    # Encode video & text of mdmmt_max, different movie here (Titanic)
    feat = vlm_api.encode_video(mid="Movies/114208744", scene_element=2, class_name='mdmmt_max')
    print(f"MDMMT MAX movie embedding: {feat}")
    text_feat = vlm_api.encode_text(text, class_name='mdmmt_max')
    print(f"MDMMT MEAN text embedding: {text_feat}")
    tembs = vlm_api.mdmmt_api.batch_encode_text(text)
    scores = torch.matmul(tembs, feat)
    for txt, score in zip(text, scores):
        print(score.item(), txt)
    print("----------------------")

    print("/nEncoding video and text of MDMMT_MEAN")
    # Encode video & text of mdmmt_mean, different movie here (Titanic)
    feat = vlm_api.encode_video(mid="Movies/114208744", scene_element=2, class_name='mdmmt_mean')
    print(f"MDMMT MEAN movie embedding: {feat}")
    text_feat = vlm_api.encode_text(text, class_name='mdmmt_mean')
    print(f"MDMMT MEAN text embedding: {text_feat}")
    scores = torch.matmul(tembs, feat)
    for txt, score in zip(text, scores):
        print(score.item(), txt)
    print("----------------------")


if __name__ == "__main__":
    main()