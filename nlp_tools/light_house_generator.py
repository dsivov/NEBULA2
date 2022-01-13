"""
The class is used to generate new lighthouse sentences from existing ones
"""
import amrlib


class LightHouseGenerator:
    def __init__(self):
        pass

    def generate(self, sent):
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
