from .tracking import utils as tracking_utils

# supported backends for detection models
BACKEND_TFLOW = 'tflow'
BACKEND_DETECTRON = 'detectron'

# a list of available backends in the current environment
__available_backends = []


# attempt to import tensorflow backend
try:
    from .detection_models.tflow import utils as tflow_utils
    __available_backends.append((BACKEND_TFLOW, tflow_utils))
except:
    pass


# attempt to import detectron2 backend
try:
    from .detection_models.detectron import utils as detectron_utils
    __available_backends.append((BACKEND_DETECTRON, detectron_utils))
except:
    pass


# check that some backend was loaded
if not __available_backends:
    raise ImportError('No detection models present in current python environment')


# some backends were successfully imported. set active backend and detection utilities
__active_detection_backend, detection_utils = __available_backends[0]


def active_detection_backend():
    """
    get the current active backend.
    @return: the name (str) of the current active backend.
    """
    return __active_detection_backend


def available_backends():
    """
    get all available backend names.
    @return: a list of strings contianing the names of available backends.
    """
    return [name for name, _ in __available_backends]


def backends_with_config(cfg: str):
    """
    find out which backends contain your desired configuration.
    @return: a list of strings that are the names of available backends that have the given configuration
             name within their module.
    """
    return [name for name, util in __available_backends if hasattr(util, cfg)]


def set_active_backend(backend_name: str):
    """
    set the current active backend by name.
    """
    try:
        idx = available_backends().index(backend_name)  # find backend name in available backends list

        # update global active backend params.
        global __active_detection_backend
        global detection_utils
        __active_detection_backend, detection_utils = __available_backends[idx]
    except ValueError:
        raise ValueError(f'backend name {backend_name} is unavailable in the current environment.')


def gpu_count():
    """
    Get the number of GPU's connected to the active backend.
    """
    if __active_detection_backend == BACKEND_TFLOW:
        # use tensorflow GPU check
        return len(detection_utils.tf.config.list_physical_devices('GPU'))
    elif __active_detection_backend == BACKEND_DETECTRON:
        # use torch GPU check
        return detection_utils.torch.cuda.device_count()
    
    return 0
