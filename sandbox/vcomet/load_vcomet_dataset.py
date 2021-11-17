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

class CREATE_VC_KG:
    def __init__(self):
        self.milvus_vc = MilvusAPI(
            'milvus', 'vcomet_visual_embed_vit_txt', 'nebula_visualcomet', 512)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)
        self.nre = NRE_API()
        self.db = self.nre.db
        self.gdb = self.nre.gdb
        self.clip_bench = NebulaVideoEvaluation()
        
        
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
            'filename': filename,
            'movie_id': None,
            'nebula_movie_id': None,
            'stage': None,
            'frame_number': None,
            'sentence': None         
            }

        # print(new_features.tolist()[0])
        # print(len(new_features.tolist()[0]))
        # input()
        
        self.milvus_vc.insert_vectors([new_features.tolist()[0]], [meta])
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
                print("Scene: ", scene)
                imf = []
                for fr in range(start_frame, stop_frame):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                    ret, frame_ = cap.read()  # Read the frame
                    feature_ = self._calculate_frame_features(frame_)
                    if torch.is_tensor(feature_):
                        imf.append(feature_.cpu().detach().numpy())
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


    def create_img_and_text_embeddings(self):
        vcr_path = "/dataset/vcomet/data/vcr1/vcr1images/"
        vcomet_kg = []
        texts = []
        vectors = []
        print("Loading files...")
        for f in glob.glob("data/vcomet*.json"):
                with open(f, "rb") as infile:
                    vcomet_kg = vcomet_kg + (json.load(infile))
        print("Remove duplicates....")
        for vg in vcomet_kg:
            #print(vg)
            vcr_img = vcr_path + vg['img_fn']
            vector_img = self._calculate_images_features(vcr_img)
            vectors.append(vector_img.detach().numpy())
            texts.append(vg['event'])
            texts.append(vg['place'])
            for intent in vg['intent']:
                texts.append(intent)
            for txt in texts:
                vectors.append(self.encode_text(txt))
            feature_mean = np.mean(vectors, axis=0)
            mean_t = torch.from_numpy(feature_mean)
            #print(mean_t)
            
            #vector = self._calculate_images_features(vcr_path+vc)
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
        text_token = torch.cat([clip.tokenize(text)]).to('cpu')
        return self.model.encode_text(text_token).detach().numpy()
    
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

    def test_movie(self):
        movie = 'Movies/92356045'
        stages = self.get_stages(movie)
        vc_db = self.gdb.connect_db("nebula_visualcomet")
    
        for stage in stages:
            print("Calculate scene for: ", stage['arango_id'])
            vector, vector_m = self.get_scene_vector(
                stage['full_path'], stage['scene_element'], stage['start'], stage['stop'])
            embedding_array = np.zeros((0, 512))
            embedding_array = np.append(embedding_array, vector, axis=0)
            #print(vector.tolist()[0])
            similar_nodes = self.milvus_vc.search_vector(50, vector.tolist()[0])
            #similar_nodes_m = self.milvus_vc.search_vector(5, vector_m.tolist()[0])
            
            #similar_nodes = similar_nodes + similar_nodes_m
            img_fns = []
            for node in similar_nodes:
                img_fns.append(node[1]['filename'].split("vcr1images/")[1])
            img_fns = list(dict.fromkeys(img_fns))
            proposed_events = []
            proposed_places = []
            proposed_intents = []
            proposed_afters = []
            proposed_befors = []
            print("Process similar nodes...")
            for img_fn in tqdm(img_fns):
                filter = {'img_fn': img_fn}
                #print(filter)
                results = vc_db.collection("vcomet_kg").find(filter)
                for result in results:
                    for person in [['man', 'woman'], ['woman', 'man'], ['girl', 'boy'], 
                    ['boy', 'girl'],['man', 'man'], ['woman', 'woman'],
                     ['girl', 'girl'], ['boy', 'boy'],["man"," "],["woman"," "], [" ", "woman"],[" ", "man"]]:
                        event = re.sub("\d+", person[0], result['event'], count=1)
                        event = re.sub("\d+", person[1], event)
                        place = result['place']
                        proposed_events.append(event)
                        proposed_places.append("this scene was filmed " + place + " ")
                        if "intent" in result:
                            for intent in result['intent']:
                                intent = re.sub("\d+", person[0], event, count=1)
                                intent = re.sub("\d+", person[1], event)
                                proposed_intents.append("intent to " + intent)
                            for after in result['after']:
                                after = re.sub("\d+", person[0], after, count=1)
                                after = re.sub("\d+", person[1], after)
                                proposed_afters.append("and then " + after)
                            for before in result['before']:
                                before = re.sub("\d+", person[0], before, count=1)
                                before = re.sub("\d+", person[1], before)
                                proposed_befors.append("before that they " + before)
            proposed_events = list(dict.fromkeys(proposed_events))
            proposed_places = list(dict.fromkeys(proposed_places))
            proposed_intents = list(dict.fromkeys(proposed_intents))
            proposed_afters = list(dict.fromkeys(proposed_afters))
            proposed_befors = list(dict.fromkeys(proposed_befors))
            print(len(proposed_events))
            print(len(proposed_places))
            print(len(proposed_intents))

            print("Top K Events...")
            score, events = self.get_top_k_from_proposed(20, proposed_events, embedding_array)
            # print("Top K Places...")
            # core, places = self.get_top_k_from_proposed(10, proposed_places, embedding_array)
            # print("Top K Intents...")
            # score, intents = self.get_top_k_from_proposed(10, proposed_intents, embedding_array)
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

            stories = []
            for proposed_event in events:
                for proposed_place in proposed_places:
                    #for proposed_intent in proposed_intents:
                    stories.append(proposed_place + " " + proposed_event)
            score, top_stories = self.get_top_k_from_proposed(5,stories, embedding_array)
            
            # #
            # print("Top K Stories with place")
            # for place in places:
            #     for story in top_stories:
            #         stories.append(place + " " + story)
            # score, top_stories = self.get_top_k_from_proposed(10, stories, embedding_array)

def main():
    kg = CREATE_VC_KG()
    # kg.collect_data()
    # kg.test_movie()
    kg.create_img_and_text_embeddings()
    

#zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)
if __name__ == "__main__":
    main()


