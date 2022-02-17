from transformers import BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from torch import nn
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, GPT2Tokenizer
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2Model, GPT2Config

def perp_score(sentence):
    tokenize_input = tokenizer_gpt.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer_gpt.convert_tokens_to_ids(tokenize_input)])
    loss = model_gpt(tensor_input, labels=tensor_input)[0]
    return math.exp(loss)

model_gpt = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model_gpt.eval()
tokenizer_gpt = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
perp_scores = []
print("Printing sentences of the file annotated_sentences.txt")
with open('./bart/sentence_correctness_classifier/annotated_sentences.txt') as f:
    input_txt = [line.rstrip('\n').rstrip().lstrip().replace(',', '').replace('\'','').replace('\xa0', '').lower() for line in f]
    input_txt = [x for x in input_txt if x != '']
    # [perp_scores.append(perp_score(i)) for i in input_txt]
    [print(f"{i} ### {perp_score(i)}") for i in input_txt[:200]]


perp_scores = []
print("Printing sentences of the file generated_sentences.txt")
with open('./bart/sentence_correctness_classifier/generated_sentences.txt') as f:
    input_txt = [line.rstrip('\n').rstrip().lstrip().replace(',', '').replace('\'','').replace('\xa0', '').lower() for line in f]
    input_txt = [x for x in input_txt if x != '']
    # [perp_scores.append(perp_score(i)) for i in input_txt]
    [print(f"{i} ### {perp_score(i)}") for i in input_txt[:200]]




