import os

import cv2
from PIL import Image
from .RemoteAPIUtility import RemoteAPIUtility


class VideoInfo:
    def __init__(self, video_path):
        self.video_path = video_path
        try:
            r = RemoteAPIUtility()
            remote_info = r.get_movie_info(video_path)
            scenes = r.get_scenes(video_path)
        except ValueError:
            remote_info = None
        if os.path.isfile(video_path):
            video = cv2.VideoCapture(video_path)
            self.num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(video.get(cv2.CAP_PROP_FPS))
            self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        elif remote_info:
            self.fps = remote_info['fps']
            self.width = remote_info['width']
            self.height = remote_info['height']
            self.num_frames = max(s['stop'] for s in scenes)
        
        elif os.path.isdir(video_path):
            items = os.listdir(video_path)
            frames = [item for item in items
                      if item.startswith('frame') and item.endswith('.jpg') or item.endswith('.png')]
            self.num_frames = len(frames)
            self.fps = None
            
            frame_img = Image.open(os.path.join(video_path, frames[0]))
            self.width = frame_img.width
            self.height = frame_img.height
        
        else:
            raise ValueError(f'video path "{video_path}" must either be a video file, a remote movie, or '
                              'an existing directory of frame images')

