import autotracker as at
from time import time
import sys

resolution = sys.argv[1]
batch_size = int(sys.argv[2])

# check GPU usage
print("Num GPUs Available: ", at.gpu_count())


RESOLUTIONS = {
    'YT':  (480, 360),    # Youtube low definition (360p)
    'SD':  (640,  480),   # Standard Definition (480p)
    'HD':  (1280, 720),   # High Definition (720p)
    'FHD': (1920, 1080),  # Full HD (1080p)
    'QHD': (2560, 1440),  # Quad HD (1440p / 2K)
    'UHD': (3840, 2160)   # Ultra HD (2160p / 4K)
}


model = at.detection_utils.VideoPredictor()

s = time()
preds = model.predict_video('Movies/92362573',
                            batch_size=batch_size,
                            resize=RESOLUTIONS[resolution])
e = time()
print('runtime:', e - s)
