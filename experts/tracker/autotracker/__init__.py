from .data.datasets import VideoDataset, FramesDataset
from .tracking import utils as tracking_utils

BACKEND_TFLOW = 'tflow'
BACKEND_DETECTRON = 'detectron'

__available_backends = []

try:
    from .detection_models.tflow import utils as tflow_utils
    __available_backends.append((BACKEND_TFLOW, tflow_utils))
except:
    pass

try:
    from .detection_models.detectron import utils as detectron_utils
    __available_backends.append((BACKEND_DETECTRON, detectron_utils))
except:
    pass

if not __available_backends:
    raise ImportError('No detection models present in current python environment')
__active_detection_backend, detection_utils = __available_backends[0]


def active_detection_backend():
    return __active_detection_backend


def available_backends():
    return [name for name, _ in __available_backends]


def backends_with_config(cfg: str):
    return [name for name, util in __available_backends if hasattr(util, cfg)]


def set_active_backend(backend_name: str):
    try:
        idx = available_backends().index(backend_name)

        global __active_detection_backend
        global detection_utils
        __active_detection_backend, detection_utils = __available_backends[idx]
    except ValueError:
        raise ValueError(f'backend name {backend_name} is unavailable in the current environment.')


def gpu_count():
    if __active_detection_backend == BACKEND_TFLOW:
        # use tensorflow GPU check
        return len(detection_utils.tf.config.list_physical_devices('GPU'))
    elif __active_detection_backend == BACKEND_DETECTRON:
        # use torch GPU check
        return detection_utils.torch.cuda.device_count()
    
    return 0
