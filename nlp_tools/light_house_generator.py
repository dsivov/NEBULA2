"""
The class is used to generate new lighthouse sentences from existing ones
"""
import amrlib
from nebula_api.atomic2020.comet_enrichment_api import Comet

class LightHouseGenerator:
    def __init__(self):
        self.comet = Comet("/home/migakol/deployment/data/comet-atomic_2020_BART")
        # The default subjects are the most widespread subjects and we check them regardless of other concepts
        self.default_subjects = ['man', 'men', 'woman', 'women', 'boy', 'boys', 'girl', 'girls', 'guy', 'guys']

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

    def generate_from_concepts(sel, concepts: list, attributes: list, persons: list, emb):
        """
        Given concepts, attributes, and persons, generate a variety of sentences and ccompare them with CLIP embedding
        :param conceptsf:
        :param attributes:
        :param persons:
        :param emb:
        :return:
        """

        # We start with generating simple sentences: [Subject - Verb - Object] + attributes for each one
        # Option 1 - go over all permutations of

        pass

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
