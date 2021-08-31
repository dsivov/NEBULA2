from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class GraphEncoder:
    """
    The class can create embeddings from a graph representation of a scene
    """
    def __init__(self):
        self.bert = SentenceTransformer('paraphrase-mpnet-base-v2')

        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
        # Scene element similarity threshold
        self.sim_th = 0.5

    def simple_graph_encoding(self, sent_list: list):
        """
        The simplest method - treat all sentences equally
        :param: sent_list - list of sentences
        :return:
        """
        embedding = self.bert.encode(sent_list)
        return embedding

    def encode_with_transformers(self, sent_list: list):
        # Tokenize sentences
        encoded_input = self.tokenizer(sent_list, padding=True, truncation=True, max_length=128, return_tensors='pt')

        tokens = torch.empty(1, 0, dtype=torch.int64)
        for sent in sent_list:
            encoded_input = self.tokenizer(sent, padding=True, truncation=True, max_length=128, return_tensors='pt')
            tokens = torch.cat((tokens, encoded_input['input_ids'][:, 1:-1]), dim=1)
            tokens = torch.cat((tokens, torch.tensor([102]).reshape((1, 1))), dim=1)

        encoded_input['input_ids'] = tokens
        encoded_input['attention_mask'] = torch.tensor([1] * max(tokens.shape)).reshape((1, max(tokens.shape)))

        if encoded_input['input_ids'].shape[1] > 512:
            encoded_input['input_ids'] = encoded_input['input_ids'][:, 0:512]
            encoded_input['attention_mask'] = encoded_input['attention_mask'][:, 0:512]

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        return sentence_embeddings


    def process_scene_sentences(self, scene_data):
        """
        :param scene_data: a list of scene elements data.
        :return: a paragraph representing the scene
        """
        # go over all scene elements
        paragraph = []
        for scene_element in scene_data:
            # Each scene element might have several MDF (most descriptive frames)
            # For each frame we take the best sentences.
            # If the subsequent sentences is similar to the previous, we delete it
            if len(scene_element['sentences']) <= 0:
                # No frame
                continue
            # Find start position
            start_pos = 0
            while True:
                if len(scene_element['sentences'][start_pos]) > 0:
                    break
                start_pos = start_pos + 1
                if start_pos >= len(scene_element['sentences']):
                    break
            if start_pos >= len(scene_element['sentences']):
                continue

            old_sent = scene_element['sentences'][start_pos][0]
            paragraph.append(old_sent)
            old_emb = self.simple_graph_encoding(old_sent)
            old_emb = old_emb / np.linalg.norm(old_emb)
            for k in range(start_pos + 1, len(scene_element['sentences'])):
                if len(scene_element['sentences'][k]) <= 0:
                    continue
                new_sent = scene_element['sentences'][k][0]
                new_emb = self.simple_graph_encoding(new_sent)
                new_emb = new_emb / np.linalg.norm(new_emb)
                if np.sum((old_emb * new_emb)) > self.sim_th:
                    continue
                old_sent = new_sent
                old_emb = new_emb
                paragraph.append(old_sent)

        return paragraph