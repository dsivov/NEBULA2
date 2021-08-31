import argparse

class BaseOptions:
    """
    This class defines the basic options used in all cases - in trainging, testing, saving data, etc.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        # --mode nlp_benchmark, clip_benchmark
        self.parser.add_argument('--mode', type=str, default='save_data', help='chooses what mode to run')
        self.parser.add_argument('--db_name', type=str, default='nebula_dev', help='arango DB name')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt


class TrainOptions(BaseOptions):
    """
    Options for saving data from Arango database
    """

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--stories_out', type=str, default='/home/migakol/data/stories.data',
                                 help='File where to save the stories. If '' stories are not saved')
        self.parser.add_argument('--stories_in', type=str, default='/home/migakol/data/stories.data',
                                 help='File from where we load the stories')
        self.parser.add_argument('--benchmark_out', type=str, default='/home/migakol/data/benchmark.data',
                                 help='Pickle file to save the benchmark')


