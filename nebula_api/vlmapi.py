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


class VLM_API:
    def __init__(self, model_name='mdmmt_mean'):
        self.available_class_names = ['clip_vit', 'clip_rn', 'mdmmt_max', 'mdmmt_mean', 'mdmmt_legacy']
        self.model_name = model_name
        if model_name not in self.available_class_names:
            raise Exception(f"Model name invalid. Use one of these names: {self.available_class_names}")
        if model_name == "clip_vit":
            self.clip_vit = CLIP_API('vit')
        elif model_name == "clip_rn":
            self.clip_rn = CLIP_API('rn')
        elif model_name == "mdmmt_max" or \
                model_name == "mdmmt_mean" or \
                    model_name == "mdmmt_legacy":
            self.mdmmt_api = MDMMT_API()
        self.remote_api = RemoteAPIUtility()
        self.nre = NRE_API()
        print(f"Available class names: {self.available_class_names}")
    
    def download_and_get_minfo(self, mid, to_print=False):
        # Download the video locally
        fps, url_link = self.nre.download_video_file(mid)
        movie_info = self.remote_api.get_movie_info(mid)
        fn = self.nre.temp_file
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
        if (scene_element_to_frames[1] - scene_element_to_frames[0]) < fps:
            print('WARNING: Scene element is less than a second.')
        encode_type = 'mean'
        if class_name == 'mdmmt_max':
            encode_type = 'max'
        return vggish_model, vmz_model, clip_model, model_vid, t_start, t_end, fps, encode_type


    def encode_video(self, mid, scene_element, class_name=None):
        if not class_name:
            class_name = self.model_name
        movie_info, fps, fn = self.download_and_get_minfo(mid, to_print=True)
        path = fn
        if class_name == 'mdmmt_max' or class_name == 'mdmmt_mean' or class_name == 'mdmmt_legacy':
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
        elif class_name == 'mdmmt_legacy':
            vid_embedding = self.mdmmt_api.encode_video_legacy(vggish_model, vmz_model, clip_model, model_vid, path, t_start, t_end, fps, encode_type)
        else:
            print(f"Error! Available class names: {self.available_class_names}")
            raise Exception("Class name you entered was not found.")
        return vid_embedding
    
    def encode_text(self, text, class_name=None):
        if not class_name:
            class_name = self.model_name
        if class_name == 'clip_rn':
            text_embedding = self.clip_rn.clip_batch_encode_text(text)
            text_embedding = torch.stack(text_embedding,axis=0)
        elif class_name == 'clip_vit':
            text_embedding = self.clip_vit.clip_batch_encode_text(text)
            text_embedding = torch.stack(text_embedding,axis=0)
        elif class_name == 'mdmmt_max':
            text_embedding = self.mdmmt_api.batch_encode_text(text)
        elif class_name == 'mdmmt_mean':
            text_embedding = self.mdmmt_api.batch_encode_text(text)
        elif class_name == 'mdmmt_legacy':
            text_embedding = self.mdmmt_api.batch_encode_text(text)
        else:
            print(f"Error! Available class names: {self.available_class_names}")
            raise Exception("Class name you entered was not found.")
        return text_embedding


    


def main():
    vlm_mdmmt_max = VLM_API(model_name="mdmmt_max")
    # vlm_mdmmt_mean = VLM_API(model_name="mdmmt_mean")
    # vlm_mdmmt_legacy = VLM_API(model_name="mdmmt_legacy")
    # vlm_clup_vit = VLM_API(model_name="clip_vit")
    # vlm_clip_rn = VLM_API(model_name="clip_rn")

    text = ['Dressed up men and women getting off a ship',
            'Dressed up men and women',
            'men and women',
            'Vulcano abrupted while dressed up men and women getting off a ship',
            'the thief was found when dressed up men and women getting off a ship',
            'Dressed up men and women getting off a ship, mouse went by the door',
            'Dressed up men and women getting off a ship and cat played with dog']

    print("/nEncoding video and text of MDMMT_MAX")
    # Encode video & text of mdmmt_max, different movie here (Titanic)
    feat = vlm_mdmmt_max.encode_video(mid="Movies/114206952", scene_element=1, class_name='mdmmt_max')
    print(f"MDMMT MAX movie embedding: {feat}")
    text_feat = vlm_mdmmt_max.encode_text(text, class_name='mdmmt_max')
    print(f"MDMMT MEAN text embedding: {text_feat}")
    tembs = vlm_mdmmt_max.mdmmt_api.batch_encode_text(text)
    scores = torch.matmul(tembs, feat)
    for txt, score in zip(text, scores):
        print(score.item(), txt)
    print("----------------------")

#     print("/nEncoding video and text of MDMMT_MEAN")
#     # Encode video & text of mdmmt_mean, different movie here (Titanic)
#     feat = vlm_mdmmt_mean.encode_video(mid="Movies/114206952", scene_element=1, class_name='mdmmt_mean')
#     print(f"MDMMT MEAN movie embedding: {feat}")
#     text_feat = vlm_mdmmt_mean.encode_text(text, class_name='mdmmt_mean')
#     print(f"MDMMT MEAN text embedding: {text_feat}")
#     scores = torch.matmul(tembs, feat)
#     for txt, score in zip(text, scores):
#         print(score.item(), txt)
#     print("----------------------")

#    # Encode video & text of clip_rn
#     print("Encoding video and text of CLIP_RN")
#     vlm_clip_rn.encode_video(mid="Movies/114207205", scene_element=0, class_name='clip_rn')
#     text_feat = vlm_clip_rn.encode_text(text, class_name='clip_rn')
#     print(f"Length of CLIP_RN text embeddings: {len(text_feat)}")
#     print("----------------------")

#     print("/nEncoding video and text of CLIP_VIT")
#     # Encode video & text of clip_vit
#     vlm_clup_vit.encode_video(mid="Movies/114207205", scene_element=0, class_name='clip_vit')
#     text_feat = vlm_clup_vit.encode_text(text, class_name='clip_vit')
#     print(f"Length of CLIP_VIT text embeddings: {len(text_feat)}")
#     print("----------------------")


if __name__ == "__main__":
    main()
