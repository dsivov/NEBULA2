# from sentence_transformers import SentenceTransformer, util
# from milvus_api.milvus_api import MilvusAPI
# bert = SentenceTransformer('paraphrase-mpnet-base-v2')
# corpus_embeddings = bert.encode(['boy running then watch pirates of the caribbean'], batch_size=64, normalize_embeddings=True, show_progress_bar=True)
# #print(corpus_embeddings)
# sims = MilvusAPI('milvus', 'bert_embeddings', 'nebula_dev', 384)

# search_ = sims.search_vector(20, corpus_embeddings[0].tolist())
# for distance, data in search_:
#     print(distance, " ", data)

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS


import torch
my_file = "all_sentences.txt"
all_sentences = open(my_file)

#print(STOPWORDS)
model_name = 'tuner007/pegasus_paraphrase'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
num_beams = 5
num_return_sequences = 5
for text in all_sentences:
    filtered_sentence = remove_stopwords(text.lower())
    src_text = [filtered_sentence]
    #print("ORIG: .......................", src_text)
    
    batch = tokenizer(src_text,truncation=True,padding='longest',max_length=10, return_tensors="pt").to(device)
    translated = model.generate(**batch,max_length=10,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)

    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    for text in tgt_text:
        print(text)
