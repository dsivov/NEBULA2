from transformers import BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from torch import nn
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, GPT2Tokenizer
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel

def perp_score(sentence, tokenizer_gpt, model_gpt):
    tokenize_input = tokenizer_gpt.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer_gpt.convert_tokens_to_ids(tokenize_input)])
    loss = model_gpt(tensor_input, labels=tensor_input)[0]
    return math.exp(loss)

def print_results(tokenizer_gpt, model_gpt, path):
    
    perp_scores = []
    print("Printing first 200 sentences of the file annotated_sentences.txt")
    with open(path) as f:
        input_txt = [line.rstrip('\n').rstrip().lstrip().replace(',', '').replace('\'','').replace('\xa0', '') for line in f]
        input_txt = [x for x in input_txt if x != '']
        # [perp_scores.append(perp_score(i)) for i in input_txt]
        [print(f"{i} ### {perp_score(i, tokenizer_gpt, model_gpt)}") for i in input_txt[:200]]

def main():
    ann_sen_path = './bart/sentence_correctness_classifier/annotated_sentences.txt'
    gen_sen_path = './bart/sentence_correctness_classifier/generated_sentences.txt'
    cs_sen_path = './bart/sentence_correctness_classifier/cs_test_sentences.txt'
    model_name = 'gpt2'
    if model_name == 'openai-gpt':
        model_gpt = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        model_gpt.eval()
        tokenizer_gpt = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        print_results(tokenizer_gpt, model_gpt, ann_sen_path)
        print_results(tokenizer_gpt, model_gpt, gen_sen_path)
        print_results(tokenizer_gpt, model_gpt, cs_sen_path)

    elif model_name == 'gpt2':
        model_gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        model_gpt.eval()
        tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')
        print_results(tokenizer_gpt, model_gpt, ann_sen_path)
        print_results(tokenizer_gpt, model_gpt, gen_sen_path)
        print_results(tokenizer_gpt, model_gpt, cs_sen_path)

    


if __name__ == "__main__":
    main()



