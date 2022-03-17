import json
import glob
from pathlib import Path
import os.path

from tqdm.std import tqdm
from nebula_api.milvus_api import MilvusAPI
import torch
import cv2
from PIL import Image
import clip
import numpy as np
import heapq
import re
from nebula_api.nebula_enrichment_api import NRE_API
from benchmark.clip_benchmark import NebulaVideoEvaluation 
from nebula_api.mdmmt_api.mdmmt_api import MDMMT_API
from transformers import BeitFeatureExtractor, FlaxBeitModel, BeitModel
from PIL import Image


class CREATE_VC_KG:
    def __init__(self):
        self.milvus_events = MilvusAPI(
            'milvus', 'vcomet_mdmmt_embedded_event', 'nebula_visualcomet', 1536)
        self.milvus_places = MilvusAPI(
            'milvus', 'vcomet_mdmmt_embedded_place', 'nebula_visualcomet', 1536)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)
        self.nre = NRE_API()
        self.db = self.nre.db
        self.gdb = self.nre.gdb
        self.clip_bench = NebulaVideoEvaluation()
        self.mdmmt = MDMMT_API()
        
    def _calculate_images_features(self, filename):
        frame_rgb = Image.open(filename).convert("RGB")
        #image = Image.fromarray(image)
        image = self.preprocess(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def _calculate_frame_features(self, frame):
        if frame is not None:
            image = self.preprocess(Image.fromarray(
                frame).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features
        else:
            return None

    def _add_image_features(self, new_features, filename):
        # Controls the addition of image features to the object
        # such that video mappings are provided, etc.,
        #new_features /= new_features.norm(dim=-1, keepdim=True)
        meta = {
            'filename': None,
            'movie_id': None,
            'nebula_movie_id': None,
            'stage': None,
            'frame_number': None,
            'sentence': filename         
            }

        # print(new_features.tolist())
        # print(len(new_features.tolist()[0]))
        # input()
        
        self.milvus_events.insert_vectors([new_features.tolist()[0]], [meta])
        #input()

    def get_stages(self, m):
        query_r = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}" RETURN doc'.format(
            m)
        cursor_r = self.db.aql.execute(query_r)
        stages = []
        for stage in cursor_r:
            stages.append(stage)
        return(stages)

    def get_scene_vector(self, fn, scene, start_frame, stop_frame):
        if (fn):
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            if video_file.is_file():
                cap = cv2.VideoCapture(fn)
                #print("Scene: ", scene)
                imf = []
                for fr in range(start_frame, stop_frame):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                    ret, frame_ = cap.read()  # Read the frame
                    feature_ = self._calculate_frame_features(frame_)
                    if torch.is_tensor(feature_):
                        imf.append(feature_.detach().cpu().numpy())
                    middle_frame = start_frame + \
                        ((stop_frame - start_frame) // 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                    ret, frame_m = cap.read()  # Read the frame
                    feature_m = self._calculate_frame_features(frame_m)
                if len(imf) > 0:
                    feature_mean = np.mean(imf, axis=0)
                    feature_t = torch.from_numpy(feature_mean)
                cap.release()
                cv2.destroyAllWindows()
                cap.release()
                cv2.destroyAllWindows()
                return(feature_t, feature_m)
        else:
            print("File doesn't exist: ", fn)
    
    def get_scene_vector_beit(self, fn, scene, start_frame, stop_frame):
        feature_extractor = BeitFeatureExtractor.from_pretrained(
            'microsoft/beit-base-patch16-224-pt22k-ft22k')
        model = BeitModel.from_pretrained(
            'microsoft/beit-base-patch16-224-pt22k-ft22k')
        #all_vectors = torch.empty((1, 197, 768))
        if (fn):
            video_file = Path(fn)
            file_name = fn
            print(file_name)
            if video_file.is_file():
                cap = cv2.VideoCapture(fn)
                #print("Scene: ", scene)
                imf = []
                #for fr in range(start_frame, stop_frame):
                middle_frame = start_frame + \
                    ((stop_frame - start_frame) // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                #cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                ret, frame_ = cap.read()  # Read the frame
                image = Image.fromarray(frame_)
                inputs = feature_extractor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                vector = outputs.last_hidden_state[0][0]
                #all_vectors = torch.cat((all_vectors, vector), dim=1)
                #print(all_vectors.size())
                #vector = torch.mean(vector, 0)
                print(vector.size())
                input()
        return(vector, vector)

    def insert_new_node(self,node):
        query = 'UPSERT { img_fn: @img_fn} INSERT  \
            { img_fn: @img_fn, description: @description, movie: @movie, metadata_fn: @metadata_fn, step: 1} UPDATE \
                { step: OLD.step + 1} IN  \
                RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
        bind_vars = {'img_fn': node['img_fn'],
                     'movie': node['movie'],
                     'metadata_fn': node['metadata_fn'],
                     'description': node['movie']
                     }
        cursor = self.db.aql.execute(query, bind_vars=bind_vars)
        for doc in cursor:
            doc = doc
        return(doc['doc']['_id'])

    def create_img_embeddings(self):
        vcr_path = "/dataset/vcomet/data/vcr1/vcr1images/"
        vcomet_kg = []
        vcr_files = []
        print("Loading files...")
        for f in glob.glob("data/vcomet*.json"):
            with open(f, "rb") as infile:
                vcomet_kg = vcomet_kg + (json.load(infile))
        print("Remove duplicates....")
        for vg in vcomet_kg:
            if vg['img_fn'] not in vcr_files:
                vcr_files.append(vg['img_fn'])
        vcr_files = list(dict.fromkeys(vcr_files))
        print("Insert into database....")
        for vc in vcr_files:
            print(vcr_path+vc)
            vector = self._calculate_images_features(vcr_path+vc)
            self._add_image_features(vector, vcr_path+vc)

    def create_places_embeddings_mdmmt(self):
        vc_db = self.gdb.connect_db("nebula_visualcomet")
        query = 'FOR doc IN vcomet_kg RETURN DISTINCT doc.place'
        cursor = vc_db.aql.execute(query)
        places = []
        for place in cursor:
            places.append(place)
            # print(places)
            text = "this scene was filmed " + place
            tembs_pl = self.mdmmt.batch_encode_text([text])
            self._add_image_features(tembs_pl, place)
            # vector = tembs_pl.tolist()[0]
            # print(len(vector))
            # print(vector)
            # input()
        # return(places)

    def create_event_embeddings_mdmmt(self):
        vc_db = self.gdb.connect_db("nebula_visualcomet")
        query = 'FOR doc IN vcomet_kg RETURN DISTINCT doc.event'
        cursor = vc_db.aql.execute(query)
        events = []
        for event in cursor:
            events.append(event)
            # places.append(place)
            # print(places)
            # text = "this scene was filmed " + place
        proposed_events = list(dict.fromkeys(events))
        for event in proposed_events:
            #for person in [['man', 'woman'], ['woman', 'man'], ["somebody", "somebody else"]]:
            event_ = re.sub("\d+", "man", event, count=1)
            event_ = re.sub("\d+", "somebody else", event_)
            # print(event_)
            # input()
            tembs_pl = self.mdmmt.batch_encode_text([event_])
            self._add_image_features(tembs_pl, [event, event_])

    def get_events_for_place(self, place):
        vc_db = self.gdb.connect_db("nebula_visualcomet")
        query = 'FOR doc IN vcomet_kg FILTER doc.place == "{}" RETURN doc'.format(place)
        cursor = vc_db.aql.execute(query)
        for doc in cursor:
            print(doc['event'])            

    def get_places_for_scene(self, movie):
        stages = self.get_stages(movie)
        places = []
        vectors = []
        for stage in stages:
            print("Find candidates for scene")
            #input()
            path = ""            
            mdmmt_v = self.mdmmt_video_encode(stage['start'], stage['stop'], path)
            vectors.append(mdmmt_v)
            if mdmmt_v is not None:
                vector = mdmmt_v.tolist()
                # print(vector)
                # input()
                similar_nodes = self.milvus_events.search_vector(5, vector)
                max_sim = similar_nodes[0][0]
                for node in similar_nodes:
                    if (max_sim - node[0]) > 0.05:
                        break
                    max_sim = node[0]
                    places.append(node[1]['sentence'])
                    print(node)
                similar_nodes = self.milvus_places.search_vector(5, vector)
                max_sim = similar_nodes[0][0]
                for node in similar_nodes:
                    if (max_sim - node[0]) > 0.05:
                        break
                    max_sim = node[0]
                    places.append(node[1]['sentence'])
                    print(node)
        #proposed_places = list(dict.fromkeys(places))
        # for pl in proposed_places:
        #     self.get_events_for_place(pl)
        #     input()

    def create_img_embeddings_beit(self):
        vcr_path = "/dataset/vcomet/data/vcr1/vcr1images/"
        vcomet_kg = []
        vcr_files = []
        print("Loading files...")
        for f in glob.glob("data/vcomet*.json"):
            with open(f, "rb") as infile:
                vcomet_kg = vcomet_kg + (json.load(infile))
        print("Remove duplicates....")
        for vg in vcomet_kg:
            #if vg['img_fn'] not in vcr_files:
            vcr_files.append(vg['img_fn'])
        # vcr_files = list(dict.fromkeys(vcr_files))

        # feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
        # model = FlaxBeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
        feature_extractor = BeitFeatureExtractor.from_pretrained(
            'microsoft/beit-base-patch16-224-pt22k-ft22k')
        model = BeitModel.from_pretrained(
            'microsoft/beit-base-patch16-224-pt22k-ft22k')
       
        print("Insert into database....")
        for vc in vcr_files:
            print(vcr_path+vc)
            image = Image.open(vcr_path+vc)
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            vector = outputs.last_hidden_state[0][0]
            print(vector.size())
            # vector = torch.mean(vector, 1)
            # print(vector.size())
            # input()
            # print(vector.tolist())
            # input()
            self._add_image_features(vector, vcr_path+vc)


    def create_img_and_text_embeddings(self):
        vcr_path = "/dataset/vcomet/data/vcr1/vcr1images/"
        vcomet_kg = []
        print("Loading files...")
        for f in glob.glob("data/vcomet*.json"):
                with open(f, "rb") as infile:
                    vcomet_kg = vcomet_kg + (json.load(infile))
        print("Remove duplicates....")
        for vg in vcomet_kg:
            texts = []
            vectors = []
            #print(vg)
            vcr_img = vcr_path + vg['img_fn']
            vector_img = self._calculate_images_features(vcr_img)
            vectors.append(vector_img)
            texts.append(vg['event'])
            texts.append(vg['place'])
            for intent in vg['intent']:
                texts.append(intent)
            for txt in texts:
                vectors.append(self.encode_text_gpu(txt))
            mean_t = torch.mean(torch.stack(vectors), dim=0)
            self._add_image_features(mean_t, vcr_img)

    def collect_data(self):
        vcr_path = "/dataset/vcomet/data/vcr1/vcr1images/"
        vcomet_kg = []
        vcr_kg = []
        print(self.db.name)
        #input()
        # for f in glob.glob("data/vcomet*.json"):
        #     with open(f, "rb") as infile:
        #         vcomet_kg = vcomet_kg + (json.load(infile))
        for f in glob.glob("data/vcr*.jsonl"):
            with open(f, "r") as infile:
                items = [json.loads(s) for s in infile]
                vcr_kg = vcr_kg + items

        # print(len(vcomet_kg))
        # # print(len(vcr_kg))
        # for vcomet_node in vcomet_kg:
        #     vcomet_node['dataset'] = "vcomet"
        #     self.db.insert_document(collection="vcomet_kg", document=vcomet_node)
        print(len(vcr_kg))
        # print(len(vcr_kg))
        for vcr_node in vcr_kg:
            vcr_node['dataset'] = "vcr"
            self.db.insert_document(
                collection="vcr_kg", document=vcr_node)

    def encode_text(self, text):
        text_token = torch.cat([clip.tokenize(text)]).to(self.device)
        return self.model.encode_text(text_token).detach().cpu().numpy()
   
    def encode_text_gpu(self, text):
        text_token = torch.cat([clip.tokenize(text)]).to(self.device)
        return self.model.encode_text(text_token)

    def get_text_img_score(self, text, img_emb):
        text_emb = self.encode_text(text)
        text_emb = text_emb / np.linalg.norm(text_emb)
        return np.sum((text_emb * img_emb))
    
    def get_top_k_from_proposed(self,top_k, proposed, embedding_array):
        proposed_map = {}
        candidates_text = []
        candidates_score = []
        for ev in tqdm(proposed):
                score = self.get_text_img_score(ev, embedding_array)
                proposed_map[score] = ev
        top_k = heapq.nlargest(top_k, proposed_map)
        for k in top_k:
            candidates_score.append(k)
            candidates_text.append(proposed_map[k])
            print (k, "->", proposed_map[k])
        return(candidates_score ,candidates_text)

    def mdmmt_video_encode(self, start_f, stop_f, path):
        import sys
        mdmmt = self.mdmmt
        path = '/home/dimas/mdmmt_test-master/1010_TITANIC_00_41_32_072-00_41_40_196.mp4'
        #path = '/home/dimas/git/NEBULA2/nebula_api/mdmmt_api/actioncliptest00108.mp4'
        #path = '/home/dimas/actioncliptrain00725.mp4'
        t_start = start_f//24
        t_end = stop_f//24
        if t_start < 0:
            print("t_start is negative! But assigning 0 to it.")
            t_start = 0
        if t_end < 0:
            print("t_end is negative! But assigning 1 to it.")
            t_end = 1
        if t_end < 1:
            t_end = stop_f / 24
        if t_start == t_end and t_start >= 1:
            t_start = t_start - 1
        print("Start/stop", t_start, " ", t_end)
        # if (t_end - t_start) >= 1:
        vemb = mdmmt.encode_video(
            mdmmt.vggish_model,  # adio modality
            mdmmt.vmz_model,  # video modality
            mdmmt.clip_model,  # image modality
            mdmmt.model_vid,  # aggregator
            path, t_start, t_end)
        return(vemb)
        # else:
        #     print("Stage too short")
        #     return(None)

    def test_mdmmt(self):
        movie = 'Movies/114208744'
        #movie = 'Movies/92361646'
        stages = self.get_stages(movie)
        vc_db = self.gdb.connect_db("nebula_visualcomet")
        proposed_events = []
        proposed_places = []
        proposed_intents = []
        # proposed_afters = []
        proposed_actions = []
        embedding_arrays = []
        mdmmt_vectors = []
        start_ = stages[0]['start']
        stop_ = 0
        for stage in stages:
            print("Find candidates for scene")
            #input()
            stop_ = stage['stop']
            mdmmt_v = self.mdmmt_video_encode(stage['start'], stage['stop'])
            vector, vector_m = self.get_scene_vector_beit(
                stage['full_path'], stage['scene_element'], stage['start'], stage['stop'])
           
            mdmmt_vectors.append(mdmmt_v)
            vector = vector.cpu().numpy().tolist()
            print(vector)
            similar_nodes = self.milvus_vc.search_vector(100, vector)
            
            img_fns = []
            for node in similar_nodes:
                img_fns.append(node[1]['filename'].split("vcr1images/")[1])
            img_fns = list(dict.fromkeys(img_fns))
            
            print("Process similar nodes...")
            for img_fn in tqdm(img_fns):
                filter = {'img_fn': img_fn}
                #print(filter)
                results = vc_db.collection("vcomet_kg").find(filter)
                for result in results:
                    for person in [['man', 'woman'], ['woman', 'man'], 
                        ["woman", "woman"],["man", "man"]]:
                        event = re.sub("\d+", person[0], result['event'], count=1)
                        event = re.sub("\d+", person[1], event)
                        place = result['place']
                        proposed_events.append(event)
                        proposed_places.append("this scene was filmed " + place + " ")
                        if "intent" in result:
                            for intent in result['intent']:
                                intent = re.sub("\d+", person[0], intent, count=1)
                                intent = re.sub("\d+", person[1], intent)
                                proposed_intents.append(intent)
                            for after in result['after']:
                                after = re.sub("\d+", person[0], after, count=1)
                                after = re.sub("\d+", person[1], after)
                                proposed_actions.append(after)
                            for before in result['before']:
                                before = re.sub("\d+", person[0], before, count=1)
                                before = re.sub("\d+", person[1], before)
                                proposed_actions.append(before)
        
        proposed_events = list(dict.fromkeys(proposed_events))
        proposed_places = list(dict.fromkeys(proposed_places))
        proposed_intents = list(dict.fromkeys(proposed_intents))
        # proposed_afters = list(dict.fromkeys(proposed_afters))
        # proposed_befors = list(dict.fromkeys(proposed_befors))
        proposed_actions = list(dict.fromkeys(proposed_actions))
        
        
        # mdmmt_v = self.mdmmt_video_encode(start_, stop_)
        # mdmmt_vector = mdmmt_v
        print("Match with MDMMT")
        for mdmmt_vector in tqdm(mdmmt_vectors):
            proposed_ev_map = {}
            proposed_pl_map = {}
            proposed_int_map = {}
            proposed_act_map = {}
            tembs_ev = self.mdmmt.batch_encode_text(proposed_events)
            scores = torch.matmul(tembs_ev, mdmmt_vector)
            for txt, score in tqdm(zip(proposed_events, scores)):
                proposed_ev_map[score] = txt
            
            tembs_pl = self.mdmmt.batch_encode_text(proposed_places)
            scores = torch.matmul(tembs_pl, mdmmt_vector)
            for txt, score in tqdm(zip(proposed_places, scores)):
                proposed_pl_map[score] = txt
            
            tembs_int = self.mdmmt.batch_encode_text(proposed_intents)
            scores = torch.matmul(tembs_int, mdmmt_vector)
            for txt, score in tqdm(zip(proposed_intents, scores)):
                proposed_int_map[score] = txt
            
            tembs_act = self.mdmmt.batch_encode_text(proposed_actions)
            scores = torch.matmul(tembs_act, mdmmt_vector)
            for txt, score in tqdm(zip(proposed_actions, scores)):
                proposed_act_map[score] = txt

            top_k_ev = heapq.nlargest(10, proposed_ev_map)
            for k in top_k_ev:
                print (k, "->", proposed_ev_map[k])
            top_k_pl = heapq.nlargest(5, proposed_pl_map)
            for k in top_k_pl:
                print (k, "->", proposed_pl_map[k])
            top_k_int = heapq.nlargest(5, proposed_int_map)
            for k in top_k_int:
                print (k, "->", proposed_int_map[k])
            top_k_act = heapq.nlargest(5, proposed_act_map)
            for k in top_k_act:
                print(k, "->", proposed_act_map[k])

    def test_movie(self):
        movie = 'Movies/114208744'

        stages = self.get_stages(movie)
        vc_db = self.gdb.connect_db("nebula_visualcomet")
        proposed_events = []
        proposed_places = []
        proposed_intents = []
        # proposed_afters = []
        proposed_actions = []
        embedding_arrays = []
        mdmmt_vectors = []
        for stage in stages:
            print("Find candidates for scene")
            #input()
            mdmmmt_v = self.mdmmt_video_encode(stage['start'], stage['stop'])
            vector, vector_m = self.get_scene_vector(
                stage['full_path'], stage['scene_element'], stage['start'], stage['stop'])
            embedding_array = np.zeros((0, 512))
            embedding_array = np.append(embedding_array, vector, axis=0)
            embedding_arrays.append(embedding_array)
            print(mdmmmt_v)
            print(embedding_array)
            input()
            #print(vector.tolist()[0])
            similar_nodes = self.milvus_vc.search_vector(100, vector.tolist()[0])
            #similar_nodes_m = self.milvus_vc.search_vector(50, vector_m.tolist()[0])
            
            # similar_nodes = similar_nodes + similar_nodes_m
            img_fns = []
            for node in similar_nodes:
                img_fns.append(node[1]['filename'].split("vcr1images/")[1])
            img_fns = list(dict.fromkeys(img_fns))
            
            print("Process similar nodes...")
            for img_fn in tqdm(img_fns):
                filter = {'img_fn': img_fn}
                #print(filter)
                results = vc_db.collection("vcomet_kg").find(filter)
                for result in results:
                    for person in [['man', 'woman'], ['woman', 'man'], 
                        ["woman", "woman"],["man", "man"]]:
                        event = re.sub("\d+", person[0], result['event'], count=1)
                        event = re.sub("\d+", person[1], event)
                        place = result['place']
                        proposed_events.append(event)
                        proposed_places.append("this scene was filmed " + place + " ")
                        if "intent" in result:
                            for intent in result['intent']:
                                intent = re.sub("\d+", person[0], intent, count=1)
                                intent = re.sub("\d+", person[1], intent)
                                proposed_intents.append(intent)
                            for after in result['after']:
                                after = re.sub("\d+", person[0], after, count=1)
                                after = re.sub("\d+", person[1], after)
                                proposed_actions.append(after)
                            for before in result['before']:
                                before = re.sub("\d+", person[0], before, count=1)
                                before = re.sub("\d+", person[1], before)
                                proposed_actions.append(before)
        
        proposed_events = list(dict.fromkeys(proposed_events))
        proposed_places = list(dict.fromkeys(proposed_places))
        proposed_intents = list(dict.fromkeys(proposed_intents))
        # proposed_afters = list(dict.fromkeys(proposed_afters))
        # proposed_befors = list(dict.fromkeys(proposed_befors))
        proposed_actions = list(dict.fromkeys(proposed_actions))
        print("Finding Top K for all scenes")
        for i, embedding_array in enumerate(embedding_arrays):
            print ("Scene: ", i)
            print("Top K Events...")
            score, events = self.get_top_k_from_proposed(10, proposed_events, embedding_array)
            print("Top K Places...")
            core, places = self.get_top_k_from_proposed(10, proposed_places, embedding_array)
            print("Top K Intents...")
            score, intents = self.get_top_k_from_proposed(10, proposed_intents, embedding_array)
            print("Top K Actions...")
            score, actions = self.get_top_k_from_proposed(10, proposed_actions, embedding_array)
                # print("Top K afters...")
                # score, intents = self.get_top_k_from_proposed(10, proposed_afters, embedding_array)
                # print("Top K Before...")
                # score, befors = self.get_top_k_from_proposed(10, proposed_befors, embedding_array)
                # print("Top K Afters...")
                # score, afters = self.get_top_k_from_proposed(10, proposed_afters, embedding_array)
                # stories = []
                # #for place in places:
                # print("Top K Stories")
                # for event in events:
                #     for intent in intents:
                #         stories.append(event + " " + intent )

                # stories = []
                # for proposed_event in proposed_events:
                #     for proposed_place in proposed_places:
                        #for proposed_intent in proposed_intents:
                #         stories.append(proposed_place + " " + proposed_event)
                # score, top_stories = self.get_top_k_from_proposed(5,stories, embedding_array)
                
                # #
                # print("Top K Stories with place")
                # for place in places:
                #     for story in top_stories:
                #         stories.append(place + " " + story)
                # score, top_stories = self.get_top_k_from_proposed(10, stories, embedding_array)

def main():
    kg = CREATE_VC_KG()
    #kg.mdmmt_video_encode()
    
    # kg.collect_data()
    # kg.test_mdmmt()
    # kg.create_img_embeddings_beit()
    # kg.create_places_embeddings_mdmmt()
    #movie = 'Movies/92361646'
    movie = 'Movies/114208744'
    #movie = 'Movies/92360929'

    kg.get_places_for_scene(movie)
    #kg.create_event_embeddings_mdmmt()

#zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)
if __name__ == "__main__":
    main()


