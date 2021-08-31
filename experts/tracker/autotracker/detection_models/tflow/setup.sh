# follow instructions from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
# tutorial: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb

# install tensorflow
echo "===== installing tensorflow ====="
pip install tensorflow
pip install tf_slim

# install coco tools
echo "===== installing COCO tools ====="
pip install pycocotools

# get models repo
echo "===== cloning models repository ====="
git clone --depth 1 https://github.com/tensorflow/models

# get protoc 3.3 to fix issue stated here: https://github.com/tensorflow/models/issues/1834
mkdir protoc_3.3
pushd protoc_3.3
wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
chmod 775 protoc-3.3.0-linux-x86_64.zip
unzip -o protoc-3.3.0-linux-x86_64.zip
popd

# install object_detection API
echo "===== installing object detection API ====="
pushd models/research/
# using previously downloaded protoc 3.3
../../protoc_3.3/bin/protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

# downgrade to tensorflow 2.4 (CUDA issue)
pip install tensorflow==2.4

# test setup
echo "===== testing setup ====="
python object_detection/builders/model_builder_tf2_test.py
popd
