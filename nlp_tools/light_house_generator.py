"""
The class is used to generate new lighthouse sentences from existing ones
"""
import amrlib
import penman
from nebula_api.atomic2020.comet_enrichment_api import Comet
from benchmark.clip_benchmark import NebulaVideoEvaluation
import numpy as np

class AMRGenerator:
    def __init__(self):
        self.v = ''
        self.s = ''
        self.o = ''
        self.l = ''

    def add_verb(self, v):
        self.v = 't / ' + v + '-01'

    def add_subj(self, s, atr_list=[], q=0):
        self.s = ':ARG0 (w / ' + s
        if q > 0:
            self.s = self.s + ' :quant ' + str(q)
        for att in atr_list:
            self.s = self.s + ':ARG1-of (s / ' + att + '-02)'
        self.s = self.s + ')'

    def add_obj(self, o, atr_list=[], q=0):
        self.o = ':ARG2 (w / ' + o
        if q > 0:
            self.o = self.o + ' :quant ' + str(q)
        for att in atr_list:
            self.o = self.o + ':mod (u / ' + att + ')'
        self.o = self.o + ')'

    def add_loc(self, l, atr_list=[]):
        self.l = ':location (a / ' + l

        for att in atr_list:
            self.l = self.l + ':mod (b / ' + att + ')'

        self.l = self.l + ')'

    def build_graph(self):
        return [self.v + ' ' + self.s + ' ' + self.o + ' ' + self.l]


class LightHouseGenerator:
    def __init__(self):
        self.comet = Comet("/home/migakol/data/comet/comet-atomic_2020_BART")
        # The default subjects are the most widespread subjects and we check them regardless of other concepts
        self.default_subjects = ['man', 'woman', 'boy', 'girl', 'guy', 'crowd', 'people']

        self.stog = amrlib.load_stog_model(model_dir='/home/migakol/data/amrlib/model_stog')
        self.gtos = amrlib.load_gtos_model(model_dir='/home/migakol/data/amrlib/model_gtos')
        self.clip_bench = NebulaVideoEvaluation()


    def decompose_lighthouse(self, events: list, actions: list, places: list):
        """
        Decompose light house events, places, and actions into components
        :param events:
        :param actions:
        :param places:
        :param emb:
        :return:
        """
        concepts = self.comet.get_groundings(events, places, 'concepts')
        attributes = self.comet.get_groundings(events, places, 'attributes')
        persons = self.comet.get_groundings(events, places, 'person', 'somebody')
        triplets = self.comet.get_groundings(events, places, 'triplet')
        verbs = self.comet.get_verbs(events)
        return concepts, attributes, persons, triplets, verbs

    def generate_from_concepts(self, concepts: list, attributes: list, persons: list, triplets: list, verbs: list,
                               places: list, emb):
        """
        Given concepts, attributes, and persons, generate a variety of sentences and ccompare them with CLIP embedding
        :param conceptsf:
        :param attributes:
        :param persons:
        :param emb:
        :return:
        """
        # We start with generating simple sentences: [Subject - Verb - Object] + attributes for each one
        # Option 1 - go over all triplets
        best_res = 0
        best_sent = ''
        # for t in triplets:
        #     sent_emb = self.clip_bench.encode_text(t)
        #     sent_emb = sent_emb / np.sum(np.linalg.norm(sent_emb))
        #     res = np.sum((emb * sent_emb))
        #     if res > best_res:
        #         best_res = res
        #         best_sent = t

        # Option 1 - go over all permutations of concepts and verbs
        # for c1 in concepts:
        #     for c2 in concepts:
        #         for v in verbs:
        #             sent = c1 + ' ' + v + c2
        #             sent_emb = self.clip_bench.encode_text(sent)
        #             sent_emb = sent_emb / np.linalg.norm(sent_emb)
        #             res = np.sum((emb * sent_emb))
        #             if res > best_res:
        #                 best_res = res
        #                 best_sent = sent

        # gen = AMRGenerator()
        # for c1 in self.default_subjects:
        #     for c2 in concepts:
        #         for v in verbs:
        #             for p in places:
        #                 gen.add_obj(c2)
        #                 gen.add_subj(c1)
        #                 gen.add_loc(p)
        #                 gen.add_verb(v)
        #
        #                 graph = gen.build_graph()
        #                 sent, _ = self.gtos.generate(graph)
        #                 sent_emb = self.clip_bench.encode_text(sent)
        #                 sent_emb = sent_emb / np.linalg.norm(sent_emb)
        #                 res = np.sum((emb * sent_emb))
        #
        #                 if res > best_res:
        #                     best_res = res
        #                     best_sent = sent

        for c1 in self.default_subjects:
            for c2 in concepts:
                for v in verbs:
                    for p in places:
                        for a1 in attributes:
                            for a2 in attributes:
                                sent = a1 + ' ' + c1 + ' ' + v + ' ' + a2 + ' ' + c2 + ' ' + p
                                if len(sent) > 320:
                                    sent = sent[0:320]
                                sent_emb = self.clip_bench.encode_text(sent)
                                sent_emb = sent_emb / np.linalg.norm(sent_emb)
                                res = np.sum((emb * sent_emb))

                                if res > best_res:
                                    best_res = res
                                    best_sent = sent

        return best_sent, best_res

if __name__ == '__main__':
    print('Start generatiion')
    stog = amrlib.load_stog_model(model_dir='/home/migakol/data/amrlib/model_stog')
    graphs = stog.parse_sents(['Crowd waits for a white steamer ship at a pier at night.'])
    for graph in graphs:
        print(graph)

    gtos = amrlib.load_gtos_model(model_dir='/home/migakol/data/amrlib/model_gtos')
    sents, _ = gtos.generate(graphs)
    for sent in sents:
        print(sent)

    print('Done generatiion')

