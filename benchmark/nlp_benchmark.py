from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from semantic_text_similarity.models import WebBertSimilarity

import numpy as np
import pandas as pd
import csv

from arango import ArangoClient

class NebulaStoryEvaluation:
    def __init__(self):
        self.tokenizer = PTBTokenizer()
        self.web_model = WebBertSimilarity(device='cpu', batch_size=10)  # defaults to GPU prediction
        self.test_set = []

    def compare_two_stories(self, story1: str, story2: str, method: str):

        if method == 'WebBert':
            sc = self.web_model.predict([(story1, story2)])
            sc = sc[0]
        else:

            gts0 = {0: [story1]}
            gts1 = {0: [story2]}

            # token1 = self.tokenizer.tokenize(gts0)
            # token2 = self.tokenizer.tokenize(gts1)

            if method == 'Bleu':
                scorer = Bleu()
            elif method == 'Meteor':
                scorer = Meteor()
            elif method == 'Rouge':
                scorer = Rouge()
            elif method == 'Cider':
                scorer = Cider()

            sc = scorer.compute_score(gts0, gts1)
            if type(sc[0]) == list:
                sc = sc[0][0]
            else:
                sc = sc[0]

        return sc


    def build_similarity_matrices(self, input_stories_full: list) -> np.array:
        """
        :param input_stories: list where each member of type gensim.models.doc2vec.TaggedDocument
            input_stories[i].words is a list of words
        :param output_file:
        :return:
        """

        np.random.seed(777)
        chosen_stories = np.random.permutation(len(input_stories_full))[0:200]

        input_stories = [input_stories_full[k] for k in chosen_stories]

        sim_matrix = []
        methods = ['Bleu', 'Meteor', 'Rouge', 'Cider', 'WebBert']
        for k in range(len(methods)):
            sim_matrix.append(np.zeros((len(input_stories), len(input_stories))))

        for k in range(len(input_stories)-1):
            for m in range(k + 1, len(input_stories)):
                for cnt, method in enumerate(methods):
                    sc = self.compare_two_stories(story1=input_stories[k], story2=input_stories[m],method=method)
                    sim_matrix[cnt][k, m] = sc

        return sim_matrix, chosen_stories


if __name__ == '__main__':

    pass
