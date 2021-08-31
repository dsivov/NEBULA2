from arango.exceptions import IndexListError
from nebula_api.milvus_api import MilvusAPI
import os
import clip
import torch


my_file = open('data/all_sentences.txt', "r")
all_sentences = my_file.read().splitlines()
#print(content_list)
milvus = MilvusAPI('sentences', 'nebula_dev')
milvus.drop_database()
metadata = []
embeddings = []
print("Read Done....")
for sentence in all_sentences:
    meta = {
                    'filename': 'none',
                    'movie_id':'none',
                    'nebula_movie_id': 'none',
                    'stage': 'none',
                    'frame_number': 'none',
                    'sentence': sentence,
            }
    metadata.append(meta)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_inputs = torch.cat([clip.tokenize(sentence)]).to(device)
    
    model, preprocess = clip.load(os.getenv('MODEL'), device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
            #print(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        #print(text_features.tolist()[0])
    embeddings.append(text_features.tolist()[0])
print("Done embedding creation, insert vector to db")
#print(metadata)
milvus.insert_vectors(embeddings, metadata)