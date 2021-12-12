import os
import sys

from . import STEP
sys.path.append(os.path.join(os.path.dirname(__file__), 'STEP'))

from .download_from_google_drive import download_file_from_google_drive
from .STEPDetector import STEPDetector, STEP_PRETRAINED_MODEL_PATH
from .customized_datasets import CustomizedMultiMovieFolder, CustomizedFrameImagesFolder, CustomizedVideoFile

STEP_PRETRAINED_MODEL_GDRIVE_ID = '1hIzrTzR50pYwLLzu_5GpmEGY4Q-e1-BX'

def download_pretrained_model(show_pbar=True):
    download_file_from_google_drive(file_id=STEP_PRETRAINED_MODEL_GDRIVE_ID,
                                    destination=STEP_PRETRAINED_MODEL_PATH,
                                    show_pbar=show_pbar)
