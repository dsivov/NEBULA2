import clip
import torch
import cv2

import numpy as np

from pathlib import Path
from PIL import Image
from nebula_api.nebula_enrichment_api import NRE_API
from experts.common.RemoteAPIUtility import RemoteAPIUtility

class CLIP_API:
    def __init__(self, vlm_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(clip.available_models())
        if vlm_name == 'rn':
            self.model, self.preprocess = clip.load("RN50x64", self.device) 
        if vlm_name == 'vit':
            self.model, self.preprocess = clip.load("ViT-L/14", self.device)
        #if vlm_name == 'vid':
        self.nre = NRE_API()
        self.db = self.nre.db
    
    def _calculate_images_features(self, frame):
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #image = Image.fromarray(image)
            image = self.preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
            return image_features
        else: 
            return None

    def clip_encode_frame(self, fn, movie_id, scene_element):
        if (fn):
            remote_api = RemoteAPIUtility()
            metadata = remote_api.get_movie_info(movie_id)
            mdfs = metadata['mdfs'][scene_element]
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            if video_file.is_file():
                cap = cv2.VideoCapture(fn)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mdfs[2])
                ret, frame_ = cap.read() # Read the frame
                if not ret:
                        print("File not found")
                else:
                    feature_t = self._calculate_images_features(frame_)
                    feature_t /= feature_t.norm(dim=-1, keepdim=True)
            return(feature_t)

    def clip_encode_video(self, fn, movie_id, scene_element):        
        if (fn):
            remote_api = RemoteAPIUtility()
            metadata = remote_api.get_movie_info(movie_id)
            mdfs = metadata['mdfs'][scene_element]
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            if video_file.is_file():
                #Simple MDF search - first, start and middle - to be enchanced 
                print("Scene: ", scene_element )
                cap = cv2.VideoCapture(fn)
                feature_mdfs = []
                for count, mdf in enumerate(mdfs):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mdf)
                    ret, frame_ = cap.read() # Read the frame
                                
                    if not ret:
                        print("File not found")
                    else:
                        feature_t = self._calculate_images_features(frame_)
                        feature_t /= feature_t.norm(dim=-1, keepdim=True)
                        feature_mdfs.append(feature_t.cpu().detach().numpy())   
                feature_mean = np.mean(feature_mdfs, axis=0)
                feature_t = torch.from_numpy(feature_mean)                
                #print(feature_t)
                return(feature_t)
                #print (start_frame)                     
        else:
            print("File doesn't exist: ", fn)
    
    def clip_encode_text(self, text):
        text_input = clip.tokenize([text]).cuda()
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        #print(text_features)
        return(text_features)
    
    def clip_batch_encode_text(self, texts):
        text_input = clip.tokenize(texts).cuda()
        batch_features = []
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
        for text_feature in text_features:
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            batch_features.append(text_feature)
        return(batch_features)

def main():
    clip=CLIP_API('rn')
    #clip.clip_encode_video('/home/dimas/0028_The_Crying_Game_00_53_53_876-00_53_55_522.mp4','Movies/114207205',0)
    clip.clip_batch_encode_text([])
if __name__ == "__main__":
    main()