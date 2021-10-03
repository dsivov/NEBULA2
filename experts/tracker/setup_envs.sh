#!/bin/bash

DETECTRON_ENV_NAME="detectron"
DETECTRON_DIR="autotracker/detection_models/detectron/"

TFLOW_ENV_NAME="tflow"
TFLOW_DIR="autotracker/detection_models/tflow/"

TRACKING_DIR="autotracker/tracking/"


function install_detectron {
    pushd $DETECTRON_DIR

    # install prerequisites
    python -m pip install "torch>=1.7.0" "torchvision>=0.8.1" "opencv-python>=4.1.2"

    # download and install detectron
    git clone https://github.com/facebookresearch/detectron2.git  #TODO fork
    python -m pip install -e detectron2

    popd  # $DETECTRON_DIR
}


function install_tflow {
    pushd $TFLOW_DIR

    # run local setup script
    ./setup.sh

    popd  # $TFLOW_DIR
}


function install_tracking {
    pushd $TRACKING_DIR

    # install dependencies
    python -m pip install "opencv-python>=4.1.2" "opencv-contrib-python>=4.1.2" shapely

    popd  # $TRACKING_DIR
}


function install_remote {
    # install dependencies
    python -m pip install python-arango boto3
}


function engines_env {
    echo "============================================="
    echo "===== Installing Detectron2 Environment ====="
    echo "============================================="
    # remove env if exists
    conda env remove -n $DETECTRON_ENV_NAME

    # install and activate env
    conda create -n $DETECTRON_ENV_NAME python=3.9 -y
    conda activate $DETECTRON_ENV_NAME
    
    # install core
    install_detectron
    install_tracking
    install_remote

    # deactivate env
    conda deactivate
    echo "============================================"
    echo "===== Detectron2 Environment Installed ====="
    echo "============================================"
}


function tflow_env {
    echo "============================================="
    echo "===== Installing Tensorflow Environment ====="
    echo "============================================="
    # remove env if exists
    conda env remove -n $TFLOW_ENV_NAME

    # install and activate env
    conda create -n $TFLOW_ENV_NAME python=3.8 -y
    conda activate $TFLOW_ENV_NAME

    # install core
    install_tflow
    install_tracking
    install_remote

    # deactivate env
    conda deactivate
    echo "============================================"
    echo "===== Tensorflow Environment Installed ====="
    echo "============================================"
}


function load_base_env {
    # load base directly
    source "$(dirname $(dirname $(which conda)))/bin/activate"

    # check for errors
    if [[ "$?" != "0" ]] || [[ $CONDA_DEFAULT_ENV != "base" ]]; then
        echo "ERROR: loading conda. please run with conda \"base\" env"
        exit 1
    fi
}


pushd "$(dirname "$0")"  # go to current script directory

# run everything from base
load_base_env

# install environments
engines_env
tflow_env

popd  # "$(dirname "$0")"
