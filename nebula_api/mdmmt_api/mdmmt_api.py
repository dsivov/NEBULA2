import warnings
warnings.filterwarnings("ignore")


import sys
import os
# try:
#     user_paths = os.environ['PYTHONPATH']
# except KeyError:
#     user_paths = "/"
# print(user_paths)   

# sys.path.append(user_paths+"/nebula_api/mdmmt_api/models/tensorflow_models/research/audioset/")
# sys.path.append(user_paths+"/nebula_api/mdmmt_api/models/tensorflow_models/research/audioset/vggish")
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/tensorflow_models/research/audioset/'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models/tensorflow_models/research/audioset/vggish'))
sys.path.append(os.path.dirname(__file__))
sys.path.append('../')

sys.path.append('/home/ilan/git/NEBULA2/nebula_api/')
import torch
torch.set_grad_enabled(False)

import scipy.io.wavfile as scio
import io
import numpy as np
from nebula_api.mdmmt_api.models.tensorflow_models.research.audioset.vggish import vggish_input as vggish_input
import subprocess

user_paths = os.path.join(os.environ['PYTHONPATH'], "nebula_api")

from dumper import ffmpeg_audio_reader
from dumper import read_frames_center_crop_batch

from models.vggish_model import VGGish
from models.vmz_model import VMZ_irCSN_152
from models.clip_model import CLIP
from models.mmt import BertTXT, BertVID

import base64
from transformers import AutoModel, AutoTokenizer 
video_id_cnt = 0   

from nebula_api import scene_detector_api

class NoAudio(Exception):
    pass

class MDMMT_API():
    def __init__(self):
        self.vggish_model = VGGish(ckpt_path=user_paths+'/mdmmt_api/ckpts/vggish_model.ckpt', per_batch_size=32)
        self.vmz_model = VMZ_irCSN_152(user_paths+'/mdmmt_api/ckpts/irCSN_152_ig65m_from_scratch_f125286141.pth')
        self.clip_model = CLIP()
        self.model_name = "bert-base-cased" 
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        state = torch.load(user_paths+'/mdmmt_api/mdmmt_model/mdmmt_3mod.pth', map_location='cpu')
        self.experts_info = {
            'VIDEO': dict(dim=2048, idx=1, max_tok=30),
            'CLIP': dict(dim=512, idx=2, max_tok=30),
            'tf_vggish': dict(dim=128, idx=3, max_tok=30),
        }
        vid_bert_params = {
            'vocab_size_or_config_json_file': 10,
            'hidden_size': 512,
            'num_hidden_layers': 9,
            'intermediate_size': 3072,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.2,
            'attention_probs_dropout_prob': 0.2,
            'max_position_embeddings': 32,
            'type_vocab_size': 19,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'num_attention_heads': 8,
        }
        model_vid = BertVID(expert_dims=self.experts_info, vid_bert_params=vid_bert_params)
        model_vid = model_vid.eval()
        model_vid.load_state_dict(state['vid_state_dict'])
        self.model_vid = model_vid.cuda()

        txt_bert_params = {
            'hidden_dropout_prob': 0.2,
            'attention_probs_dropout_prob': 0.2,
        }
        model_txt = BertTXT(
            modalities=list(self.experts_info.keys()),
            add_special_tokens=True,
            txt_bert_params=txt_bert_params,
        )
        model_txt = model_txt.eval()
        model_txt.load_state_dict(state['txt_state_dict'])
        self.model_txt = model_txt.cuda()

    def read_video_segm(self, abspath, t_beg, t_end):
        cmd = f'ffmpeg -y -ss {t_beg} -i {abspath} -max_muxing_queue_size 9999  -loglevel error -f mp4 -vf scale="(floor(112/ih * iw/2))*2:112"  -c:a copy  -movflags frag_keyframe+empty_moov -t {t_end - t_beg} pipe:1 -nostats -hide_banner -nostdin'
        p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        assert p.returncode == 0, cmd
        buf = p.stdout
        return buf

    def unpack_wav(self, wav):
        sr, data = scio.read(io.BytesIO(wav))
        data = data / 32768.0
        segms = vggish_input.waveform_to_examples(data, sr)
        t_start = np.arange(len(segms), dtype=np.float32) * 0.96
        t_end = t_start + 0.96
        timings = np.concatenate([t_start[..., None], t_end[..., None]], axis=1) # (nsegm, 2)
        return timings, segms


    def vggish_compute_embs(self, model, path, t_start, t_end, batch_size=32):
        wav = ffmpeg_audio_reader(path, t_start, t_end)
        if wav is None:
            # no audio channel
            raise NoAudio
        timings, segms = self.unpack_wav(wav)
        
        embs = []
        idxs = range(0, len(segms), batch_size)
        for idx in idxs:
            embs.append(model(segms[idx: idx + batch_size]))
        embs = np.concatenate(embs, axis=0)
        return timings, embs

    def visual_compute_embs(self,
        model,
        path,
        t_start,
        t_end,
        fps=32,
        frame_size=624,
        frame_crop_size=624,
        per_batch_size=4,
        frames_per_clip=32):
               
        frames_batch_iter = read_frames_center_crop_batch(
                    path,
                    fps=fps,
                    frame_size=frame_size,
                    frame_crop_size=frame_crop_size,
                    batch_num_frames=per_batch_size*frames_per_clip,
                    t_start=t_start,
                    t_end=t_end)
        embs = []
        timings = []
        t = 0
        delta = frames_per_clip / fps
        for frames in frames_batch_iter:
            if len(frames) % frames_per_clip > 0:
                n = len(frames)
                n1 = int(len(frames) // frames_per_clip * frames_per_clip)
                frames1 = frames[:n1]
                # increase frame rate in the last video segment
                idxs = np.ceil(np.linspace(n1, n-1, frames_per_clip)).astype(np.long)
                frames2 = frames[idxs]
                frames = np.concatenate([frames1, frames2], axis=0)
            assert len(frames) % frames_per_clip == 0
            batch_frames = frames.reshape(-1, frames_per_clip, frame_crop_size, frame_crop_size, 3)
            for _ in range(len(batch_frames)):
                timings.append((t, t + delta))
                t += delta
            
            embs.append(model(batch_frames))
       
        embs = np.concatenate(embs, axis=0)
        timings = np.array(timings)
        return timings, embs


    def visual_clip_compute_embs(self,
        model,
        path,
        t_start,
        t_end,
        fps=32,
        frame_size=624,
        frame_crop_size=624,
        per_batch_size=4,
        frames_per_clip=32):
        embs = []
        timings = []
        t = 0
        frames = scene_detector_api.divide_movie_by_timestamp(path, 3, 4.38, (frame_size, frame_size))
        frames = np.array(frames) # must be 32 frames
        delta = len(frames) / fps
        for idx in range(len(frames)):
            timings.append((t, t + delta))
            t += delta
        
        for idx in range(8):
            batch_frames = frames[idx*4:(idx+1)*4].reshape(-1, frames_per_clip, frame_crop_size, frame_crop_size, 3)
            embs.append(model(batch_frames))
       
        embs = np.concatenate(embs, axis=0)
        timings = np.array(timings) # (nsegm, 2)
        return timings, embs

    def prepare_features(self, features, features_t):
        all_features = {}
        all_features_t = {}
        all_features_mask = {}

        for mod_name, einfo in self.experts_info.items():
            max_tok = einfo["max_tok"]
            dim = einfo["dim"]
            all_features[mod_name] = torch.zeros(1, max_tok, dim)
            all_features_t[mod_name] = torch.zeros(1, max_tok)
            all_features_mask[mod_name] = torch.zeros(1, max_tok)
        for mod_name in features.keys():
            max_tok = self.experts_info[mod_name]["max_tok"]
            mod_feat = features[mod_name]
            mod_feat_t = np.array(features_t[mod_name])
            if mod_feat is None:
                continue
            assert len(mod_feat) == len(mod_feat_t), (len(mod_feat), len(mod_feat_t))
            if np.isnan(mod_feat_t.sum()):
                mod_feat_t = np.zeros(len(mod_feat_t))
                mod_feat_t[:] = 1
            else:
                mod_feat_t = mod_feat_t - mod_feat_t[:,0].min()
                mod_feat_t = 2 + (mod_feat_t[:,1] + mod_feat_t[:,0]) / 2 # (ntok,)
            all_features[mod_name][0,:len(mod_feat)] = torch.from_numpy(mod_feat[:max_tok].copy())
            all_features_t[mod_name][0,:len(mod_feat)] = torch.from_numpy(mod_feat_t[:max_tok].copy())
            all_features_mask[mod_name][0, :len(mod_feat)] = 1
        return all_features, all_features_t, all_features_mask

    def dict_to_cuda(self, d):
        return {k:v.cuda() for k, v in d.items()}


    def encode_video(self, vggish_model, vmz_model, clip_model, model_vid, path, t_start=None, t_end=None):
        try:
            timings_vggish, embs_vggish = self.vggish_compute_embs(vggish_model, path, t_start, t_end)
        except NoAudio:
            timings_vggish, embs_vggish = None, None
        timings_vmz, embs_vmz = self.visual_compute_embs(vmz_model, path, t_start, t_end,
                                                    fps=28, frames_per_clip=32, frame_crop_size=224, frame_size=224)
        timings_clip, embs_clip = self.visual_clip_compute_embs(clip_model, path, t_start, t_end,
                                                    fps=28, frames_per_clip=1, frame_crop_size=224, frame_size=224)


        features = {
            'VIDEO': embs_vmz,
            'CLIP': embs_clip,
            'tf_vggish': embs_vggish,
        }

        features_t = {
            'VIDEO': timings_vmz,
            'CLIP': timings_clip,
            'tf_vggish': timings_vggish,
        }
        
        all_features, all_features_t, all_features_mask = self.prepare_features(features, features_t)
        
        all_features = self.dict_to_cuda(all_features)
        all_features_t = self.dict_to_cuda(all_features_t)
        all_features_mask = self.dict_to_cuda(all_features_mask)
        
        out = model_vid(all_features, all_features_t, all_features_mask) # (1, 512*3)
        #output.append(out[0])
        return out[0]#torch.max(torch.stack(output), dim=0) # output

    def encode_text(self, text):
        emb = self.model_txt([text])[0]
        return emb

    def batch_encode_text(self, texts):
        embs = self.model_txt(texts)
        return embs

    def sim(x1, x2):
        return (x1*x2).sum()

def main():
    mdmmt = MDMMT_API()
    path = '/dataset/development/1010_TITANIC_00_41_32_072-00_41_40_196.mp4'
    t_start=3
    t_end=4
    vemb = mdmmt.encode_video(
        mdmmt.vggish_model, # adio modality
        mdmmt.vmz_model, # video modality
        mdmmt.clip_model, # image modality
        mdmmt.model_vid, # aggregator
        path, t_start, t_end)
    texts = [
        'actor stands closely behind a red haired woman',
        'this scene was filmed on a cathedral balcony',
        'a man in t-shirt sits near the computer',
        'a man in shirt sits near the computer',
        'a man in a shirt sits in front of a computer',
        'a man in a t-shirt sits in front of a computer',
        'woman is resting her hand on the ship rail and conversing with someone',
        'a women is standing near men on the boat',
        'woman is dressed in finery and hangs on the edge of the boat looking sad',
        'a man is walking',
        'this scene is on a large boat',
        'man in red jacket',
        'a man walks by a chair',
        'A man is jumping near the chair',
        'A man is jumping',
        'A man walks',
        'this scene was filmed on a ship deck',
        'the man jumps'
    ]
    tembs = mdmmt.batch_encode_text(texts)
    scores = torch.matmul(tembs, vemb)
    for txt, score in zip(texts, scores):
        print(score.item(), txt)
    # tembs = mdmmt.batch_encode_text(texts)
    # scores_mat = np.zeros((len(vemb), len(texts)))
    # for idx, out in enumerate(vemb):
    #     scores = torch.matmul(tembs, out)
    #     scores_mat[idx] = scores.cpu().detach().numpy() 

    # for idx, txt in enumerate(texts):
    #     max_score = np.max(scores_mat[:, idx])
    #     avg_score = np.average(scores_mat[:, idx])
    #     print(f" Max: {max_score:.3f}, Avg: {avg_score:.3f} {txt}")

if __name__ == "__main__":
    main()
