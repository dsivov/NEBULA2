import os
import argparse
import subprocess
import threading
import queue

from tqdm import tqdm
import numpy as np
import torch
import cv2

from mp_utils import MpGen


import multiprocessing as mp
import signal
import traceback
import sys
import PIL.Image
import PIL


def proc_pack(input_it, dst_prefix):
    print('Started proc_pack')
    emb_fname = dst_prefix+'.emb'
    idx_fname = dst_prefix+'.idx'
    dname = os.path.dirname(emb_fname)
    os.makedirs(dname, exist_ok=True)
    with open(emb_fname, 'wb') as fout_emb, open(idx_fname, 'w') as fout_idx:
        for embs, paths in input_it:
            fout_emb.write(embs.tobytes())
            for path in paths:
                fout_idx.write(f'VIDEO\t{path}\n')
                fout_idx.write(f'0\t1\n')
                yield path


def center_crop(img):
    h, w, c = img.shape
    assert c==3, img.shape
    m = min(h, w)
    return img[int(h/2-m/2):int(h/2+m/2), int(w/2-m/2):int(w/2+m/2), :]



class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, input_it, bs, frame_size, mode='image', frames_per_clip=None):
        self.input_it = input_it
        self.bs = bs
        self.frame_size = frame_size
        self.mode = mode
        self.frames_per_clip = frames_per_clip


    def __iter__(self):
        return self

    def __next__(self):
        # decode
        imgs = []
        paths = []
        for path in self.input_it:
            try:
                img = np.asarray(PIL.Image.open(path).convert('RGB'))
            except Exception:
                continue
            if len(img.shape) == 2:
                img = img[..., None].repeat(3, axis=2)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = img.repeat(3, axis=2)
            #img = cv2.imread(path) # h,w,c
            #img = img[:,:,::-1] # bgr 2 rgb
            img = center_crop(img)
            img = cv2.resize(img, (self.frame_size, self.frame_size))
            img = img[None, ...] # (1,h,w,c)
            imgs.append(img)
            paths.append(path)
            if len(paths) == self.bs:
                break
        if len(paths) == 0:
            raise StopIteration
        imgs = np.concatenate(imgs, axis=0).astype(np.float32)
        if self.mode == 'image':
            imgs = np.expand_dims(imgs, 1) # (bs, 1, h, w, c)
        elif self.mode == 'moving_picture':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return (imgs, paths)


def proc_dumper_1(
        input_it,
        frame_size,
        #frame_crop_size,
        frames_per_clip,
        per_batch_size,
        model,
        lock,
        q_out,
        num_workers):
    dataset = MyIterableDataset(
            input_it=input_it,
            bs=32,
            frame_size=frame_size,
            frames_per_clip=frames_per_clip)
    loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=lambda items: items[0],
            batch_size=1)
    for imgs, paths in loader:
        #imgs: (bs, 1, h, w, c) if mode == 'image'
        #imgs: (bs, t, h, w, c) if mode == 'moving_picture'
        with lock:
            embs = model(imgs)
        q_out.put((embs, paths))
    q_out.put(None)

    
def proc_dumper(
        input_it,
        gpu,
        frame_size,
        #frame_crop_size,
        frames_per_clip,
        model_type,
        per_batch_size,
        num_readers=2,
        num_workers=8):
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
    if model_type == 'VMZ_irCSN_152':
        from models.vmz_model import VMZ_irCSN_152
        model = VMZ_irCSN_152('ckpts/irCSN_152_ig65m_from_scratch_f125286141.pth')
    elif model_type == 'CLIP':
        from models.clip_model import CLIP
        model = CLIP()
    else:
        raise NotImplementedError
    lock = threading.Lock()
    q = queue.Queue(20)
    threads = []
    for _ in range(num_readers):
        th = threading.Thread(target=proc_dumper_1, kwargs=dict(
            input_it=input_it,
            frame_size=frame_size,
            #frame_crop_size=frame_crop_size,
            frames_per_clip=frames_per_clip,
            per_batch_size=per_batch_size,
            model=model,
            lock=lock,
            q_out=q,
            num_workers=num_workers))
        th.start()
        threads.append(th)
    num_alive = num_readers
    while num_alive > 0:
        x = q.get()
        if x is None:
            num_alive -= 1
            continue
        yield x
    for th in threads:
        th.join()





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, choices=['CLIP', 'VMZ_irCSN_152'])
    parser.add_argument('--gpus', type=lambda x: list(map(int, x.split(','))), default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--dst_prefix', required=True)
    parser.add_argument('--lst', help='each line is path to video file', required=True)
    parser.add_argument('--nworker_per_gpu', type=int, default=4)
    parser.add_argument('--num_readers', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--per_batch_size', type=int, default=8)
    parser.add_argument('--frame_size', type=int, default=256)
    #parser.add_argument('--frame_crop_size', type=int, default=224)
    parser.add_argument('--frames_per_clip', type=int, default=1)
    args = parser.parse_args()

    lst = [] # paths to video files
    with open(args.lst) as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            lst.append(path)

    g0 = lst
    
    num_workers_dumper = len(args.gpus)*args.nworker_per_gpu
    if args.model_type in ['CLIP', 'VMZ_irCSN_152']:
        proc_dumper_fn = lambda input_it, rank: proc_dumper(
                input_it=input_it,
                gpu=args.gpus[rank % len(args.gpus)],
                frame_size=args.frame_size,
                #frame_crop_size=args.frame_crop_size,
                frames_per_clip=args.frames_per_clip,
                model_type=args.model_type,
                per_batch_size=args.per_batch_size,
                num_readers=args.num_readers,
                num_workers=args.num_workers)
    g1 = MpGen(g0,
            proc_dumper_fn,
            num_workers=num_workers_dumper,
            streaming_mode=True)

    proc_pack_fn = lambda input_it, rank: proc_pack(
            input_it=input_it,
            dst_prefix=args.dst_prefix)    
    g2 = MpGen(g1, proc_pack_fn, num_workers=1, streaming_mode=True)

    for _ in tqdm(g2, total=len(g0)):
        pass



if __name__ == "__main__":
    main()
