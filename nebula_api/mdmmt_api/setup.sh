imkdir -p ckpts
# https://github.com/facebookresearch/VMZ/
wget https://github.com/bjuncek/VMZ/releases/download/test_models/irCSN_152_ig65m_from_scratch_f125286141.pth -O ckpts/irCSN_152_ig65m_from_scratch_f125286141.pth

mkdir ckpts
# https://github.com/tensorflow/models/tree/master/research/audioset/vggish
wget https://storage.googleapis.com/audioset/vggish_model.ckpt -O ckpts/vggish_model.ckpt

git clone https://github.com/openai/CLIP models/CLIP
git clone https://github.com/tensorflow/models/ models/tensorflow_models

mkdir mdmmt_model

