#@title Model
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch import nn
from typing import Tuple, List, Union, Optional
import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image
import clip
import cv2 as cv
import os

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


D = torch.device
CPU = torch.device('cpu')

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                # logits = logits / (temperature if temperature > 0 else 1.0)
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                # pp = torch.argsort(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

def generate_with_previous(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
        prev_sent = ''
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    good_tokens = [13, 32, 64, 257, 262, 286, 287, 319, 340, 351, 379, 464, 818, 1290, 1474, 2029, 2157, 2166, 2174,
                   2202, 2953, 21106, 21428, 32397, 34163, 40640]

    prev_tokens = tokenizer.encode(prev_sent)
    prev_tokens = list(set(prev_tokens) - set(good_tokens))

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                # logits = logits / (temperature if temperature > 0 else 1.0)
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits[0, prev_tokens] = -np.inf
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                # pp = torch.argsort(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


class ClipCap:
    def __init__(self, is_coco=True):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.prefix_length = 10
        if is_coco:
            self.model_path = '/home/ilan/git/NEBULA2-latest/NEBULA2/pipeline/weights/coco_weights.pt'
            self.model_path = '/home/migakol/data/clip_cap/coco_weights.pt'
        else:
            self.model_path = '/home/ilan/git/NEBULA2-latest/NEBULA2/pipeline/weights/conceptual_weights.pt'
            self.model_path = '/home/migakol/data/clip_cap/conceptual_weights.pt'


        self.model = ClipCaptionModel(self.prefix_length)

        self.model.load_state_dict(torch.load(self.model_path, map_location=CPU))
        self.model = self.model.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def generate_text(self, emb, use_beam_search=False, num_versions=0):
        """
        :param emb: - clip embedding
        :return:
        """
        prefix_embed = self.model.clip_project(torch.tensor(emb, dtype=torch.float32)).reshape(1, self.prefix_length,-1)
         
        if use_beam_search:
            # generated_text_prefix = generate_beam(self.model, self.tokenizer, embed=prefix_embed, beam_size=5)
            generated_text_prefix = generate_beam(self.model, self.tokenizer, embed=prefix_embed, beam_size=50)
        else:                                                                                       
            generated_text_prefix = generate2(self.model, self.tokenizer, embed=prefix_embed)
            old_text = generated_text_prefix
            if num_versions > 0:
                ret_text = [''] * num_versions
                ret_text[0] = generated_text_prefix
            for k in range(num_versions-1):
                new_text = generate_with_previous(self.model, self.tokenizer, embed=prefix_embed,
                                              prev_sent=old_text)
                ret_text[k+1] = new_text
                old_text = old_text + ' ' + new_text
            if num_versions > 0:
                generated_text_prefix = ret_text

        return generated_text_prefix


from nebula_api.atomic2020.comet_enrichment_api import Comet
import csv
def get_clipcap_examples():
    comet = Comet("/home/migakol/data/comet/comet-atomic_2020_BART")

    query = 'FOR doc IN nebula_clipcap_results RETURN doc'
    cursor = comet.db.aql.execute(query, ttl=3600)

    movies_list = []
    for cnt, movie_data in enumerate(cursor):
        movies_list.append(movie_data)

    outfolder = '/home/migakol/data'
    out_file = os.path.join(outfolder, 'clipcap_result_csv2.csv')

    with open(out_file, 'w', newline='') as csvfile:
        fieldnames = ['path', 'sentence0', 'sentence1', 'sentence2', 'scene_element']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'path': 'path', 'sentence0': 'sentence0', 'sentence1': 'sentence1',
                         'sentence2': 'sentence2', 'scene_element': 'scene_element'})
        for movie in movies_list:
            writer.writerow({'path': movie['path'],
                             'sentence0': movie['sentence0'],
                             'sentence1': movie['sentence1'],
                             'sentence2': movie['sentence2'],
                             'scene_element': movie['scene_element']})

    return movies_list


if __name__ == '__main__':

    get_clipcap_examples()

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load("ViT-B/32", device=device)
    clip_cap = ClipCap()
    mdfs_path = "/movies/mdfs/"
    prefix_link = "http://ec2-18-159-140-240.eu-central-1.compute.amazonaws.com:7000/static/dataset1/mdfs/"
    # images = [os.path.join(mdfs_path, image) for image in os.listdir(mdfs_path)]
    images = [' ', '']
    for img_path in images:
        img_path = '/home/migakol/data/tiktalk08.png'
        frame = cv.imread(img_path)
        img = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
        embedding = model.encode_image(img)
        output = clip_cap.generate_text(embedding, use_beam_search=True)
        image_name = img_path.split("/")[-1]
        print("Input:")
        print(image_name)
        link_path = os.path.join(prefix_link, image_name)
        print(f"Link: {link_path}")
        print("Output")
        for out in output:
            print(out)
