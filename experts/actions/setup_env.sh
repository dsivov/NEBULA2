#!/bin/bash


ENV_NAME="step"
ACTION_DETECTOR_DIR="stepwrapper"

APEX_PATH_EXPORT="export PATH=/usr/local/cuda/bin:\$PATH"
APEX_LD_LIB_EXPORT="export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"


function install_step {
    # install dependencies
    conda install ninja -y
    python -m pip install torch torchvision torchaudio opencv-python tqdm requests

    pushd $ACTION_DETECTOR_DIR

    # install apex for GPU
    git clone https://github.com/NVIDIA/apex.git  #TODO fork
    pushd apex/
    python setup.py install --cuda_ext --cpp_ext
    popd  # apex/

    # install STEP package
    git clone https://github.com/migakol/STEP.git
    python download_from_google_drive.py 1hIzrTzR50pYwLLzu_5GpmEGY4Q-e1-BX -o STEP/pretrained/ava_step.pth
    pushd STEP/
    python setup.py build develop
    python demo.py  # test STEP setup
    popd  # STEP/

    popd  # $ACTION_DETECTOR_DIR

}


function install_remote {
    # install dependencies
    python -m pip install python-arango boto3
}


function install_env {
    echo "======================================="
    echo "===== Installing STEP Environment ====="
    echo "======================================="
    # remove env if exists
    conda env remove -n $ENV_NAME

    # install and activate env
    conda create -n $ENV_NAME python=3.9 -y
    conda activate $ENV_NAME

    # install core
    install_step
    install_remote

    # deactivate env
    conda deactivate
    echo "============================================"
    echo "===== Environment Installed ====="
    echo "============================================"
}


function load_base_env {
    # load base directly
    source "$(dirname $(dirname $(which conda)))/bin/activate"

    # check for errors
    if [[ "$?" != "0" ]] || [[ $CONDA_DEFAULT_ENV != "base" ]]; then
        echo "ERROR: issue loading conda. please run with conda \"base\" env"
        exit 1
    fi
}


pushd "$(dirname "$0")"  # go to current script directory

# run everything from base
load_base_env

# install environment
install_env

popd  # "$(dirname "$0")"


# prepare environment variables in .zshrc
if grep -Fxq "$APEX_PATH_EXPORT" ~/.zshrc
then
    echo "\"$APEX_PATH_EXPORT\" already in ~/.zshrc"
else
    eval $APEX_PATH_EXPORT
    echo $APEX_PATH_EXPORT >> ~/.zshrc
    echo "added \"$APEX_PATH_EXPORT\" to ~/.zshrc"
fi

if grep -Fxq "$APEX_LD_LIB_EXPORT" ~/.zshrc
then
    eval $APEX_LD_LIB_EXPORT
    echo "\"$APEX_LD_LIB_EXPORT\" already in ~/.zshrc"
else
    echo $APEX_LD_LIB_EXPORT >> ~/.zshrc
    echo "added \"$APEX_LD_LIB_EXPORT\" to ~/.zshrc"
fi