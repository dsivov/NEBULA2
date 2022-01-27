import sys
import os
import numpy as np
import pandas as pd
import urllib
import subprocess
import re
import tempfile
import torch
from typing import List

from nebula_api.nebula_enrichment_api import *
from experts.common.RemoteAPIUtility import RemoteAPIUtility
# from nebula_api.mdmmt_api import mdmmt_api

nre = NRE_API()
api = RemoteAPIUtility()
# mdmmt = mdmmt_api.MDMMT_API()

BASE_DIR = sys.modules['nebula_api'].__spec__.submodule_search_locations[0]

def range_get_val_for_frame(frame,val,data):
    return [x[val] for x in data if x['start']<=frame and x['stop']>frame]
def tracker_get_box_for_frame(frame,data):
    return [x['bboxes'][str(frame)] for x in data if x['start']<=frame and x['stop']>frame]
def step_get_val_for_frame(frame, val,data):
    return [x[val] for x in data if int(x['frame_id'])==frame]


def normalize_movie(mid):
    action_data = nre.get_all_expert_data("Actions", mid)
    object_data = nre.get_all_expert_data("Object", mid)
    clip_data = nre.get_clip_data(mid)
    vcomet_data = nre.get_vcomet_data(mid)
    movie_info = api.get_movie_info(mid)
    frame_array = []
    for i in range(movie_info['last frame']):
        scene_element_val = range_get_val_for_frame(i,'scene_element', object_data)        
        frame_data = {'arango_id': mid, 'frame_id': i, 
                      'step_actor_id': step_get_val_for_frame(i, 'actor_id', action_data),
                      'step_description': step_get_val_for_frame(i, 'description', action_data),
                      'step_box': step_get_val_for_frame(i,'box', action_data),
                      'tracker_description': range_get_val_for_frame(i, 'description', object_data),
                      'scene_element': scene_element_val[0] if scene_element_val else None,
                      'tracker_box': tracker_get_box_for_frame(i, object_data),
                      'simulated_expert': []}                          
        scene = frame_data['scene_element']
        # Verify with dima what is order of scene in clip vs vcomet
        # frame_data['clip_text'] = None
        # frame_data['clip_triplets'] = None
        frame_data['vcomet_places'] = None
        frame_data['vcomet_events'] = None
        frame_data['vcomet_actions'] = None
        if scene is not None:
            # frame_data['clip_text'] = clip_data[scene][0]
            # frame_data['clip_triplets'] = clip_data[scene][1]
            frame_data['vcomet_places'] = range_get_val_for_frame(i,'places',vcomet_data)
            frame_data['vcomet_events'] = range_get_val_for_frame(i,'events',vcomet_data)
            frame_data['vcomet_actions'] = range_get_val_for_frame(i,'actions',vcomet_data)                    
        frame_array.append(frame_data)
    return frame_array

def create_normalized_frame(frame, experiment='default'):
    frame['experiment'] = experiment
    update_vars = ', '.join((['{}: @{}'.format(k,k) for k in frame.keys()]))
    query = 'UPSERT { arango_id: @arango_id, \
                                frame_id: @frame_id, experiment: @experiment}\
            INSERT  \
                {'+update_vars+', updates: 1}\
            UPDATE \
                { updates: OLD.updates + 1, '+update_vars+'} IN normalized_frames \
                    RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'       
    return api.db.aql.execute(query, bind_vars=frame)

def save_normalized_collection(frames, **kwargs):
    for frame in frames:
        try:
            create_normalized_frame(frame, **kwargs)
        except:
            print('Failed on {}'.format(frame['frame_id']))

def recreate_normalized_movie(mid, **kwargs):
    frames = normalize_movie(mid)
    save_normalized_collection(frames, **kwargs)

def get_normalized_frame(mid, frame, experiment='default'):
    query = "FOR doc IN normalized_frames FILTER doc.arango_id == '{}' AND doc.frame_id == {} AND doc.experiment == '{}' RETURN doc".format(mid,frame,experiment)
    print(query)
    cur = list(api.db.aql.execute(query))
    if cur:
        return cur[0]    # We only have one normalized frame (?)

def get_normalized_range(mid, start, stop, experiment='default'):
    query = "FOR doc IN normalized_frames FILTER doc.arango_id == '{}' AND doc.frame_id >= {} AND doc.frame_id < {} AND doc.experiment == '{}' RETURN doc".format(mid,start,stop,experiment)
    return list(api.db.aql.execute(query))    

def get_normalized_scene(mid, scene, experiment='default'):
    query = "FOR doc IN normalized_frames FILTER doc.arango_id == '{}' AND doc.scene_element == {} AND doc.experiment == '{}' RETURN doc".format(mid,scene,experiment)
    return list(api.db.aql.execute(query))        
def get_normalized_movie(mid, experiment='default'):
    query = "FOR doc IN normalized_frames FILTER doc.arango_id == '{}' AND doc.experiment == '{}' RETURN doc".format(mid,experiment)
    return list(api.db.aql.execute(query))
def get_or_create_normalized_video(mid, **kwargs):
    frames = get_normalized_movie(mid,**kwargs)
    if not frames:
        recreate_normalized_movie(mid, **kwargs)
        frames = get_normalized_movie(mid, **kwargs) 
    return pd.DataFrame(frames).drop(['_key','_id','_rev', 'updates'],axis=1).replace({np.nan: None})

def frame_to_concepts(frame)-> List:
    def transform_concept(c):
        exp = re.compile(r"^([a-zA-z]+)(\d*)$")
        r = exp.match(c)
        return r.group(1) if r else c
        
    pre_concepts = set(frame['tracker_description']).union(set(frame['step_description'])).union(set(frame['simulated_expert']))
    concepts = list(set(map(transform_concept,pre_concepts)))
    return concepts

def kgbart_fusion(frames) -> (List[str], List[str]):
    h, outname = tempfile.mkstemp(text=True)
    os.close(h)
    h, fname = tempfile.mkstemp(text=True)
    os.close(h)
    KGBART_MAIN = BASE_DIR+'/kgbart/KGBART/KGBART_training/decode_seq2seq.py'
    KGBART_CC_DIR = BASE_DIR+'/kgbart/downloaded/commongen_dataset'
    KGBART_MODEL_DIR = BASE_DIR+'/kgbart/output/best_model/model.best.bin'
    options = {
        'data_dir': KGBART_CC_DIR,
        'output_dir': os.path.dirname(outname),
        'input_file': fname,
        'model_recover_path': KGBART_MODEL_DIR,
        'output_file': os.path.basename(outname),
        'split': 'dev',
        'beam_size': 5,
        'forbid_duplicate_ngrams': True
    }
    all_concepts = []
    with open(fname, 'w') as f:
        for frame in frames:
            concepts = frame_to_concepts(frame)
            all_concepts.append(', '.join(concepts))
            f.write(' '.join(concepts)+'\n')
        
    # write expert tokens to input file
    
    cmdline = 'python '+KGBART_MAIN+' '+ ' '.join(['--{} {}'.format(k,v) for (k,v) in options.items()]) + '>/dev/null 2>&1'
    os.system(cmdline)
    with open(outname,'r') as f:
        rc = f.readlines()
    os.unlink(outname)
    os.unlink(fname)
    return all_concepts, rc
