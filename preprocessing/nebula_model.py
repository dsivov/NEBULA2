import random
import numpy as np
from datetime import datetime
from karateclub.estimator import Estimator
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class NEBULA_DOC_MODEL(Estimator):
    def __init__(self, algo: int=0, words=0, window: int=5, dimensions: int=128, workers: int=8,
                 epochs: int=500, down_sampling: float=0,
                 learning_rate: float=0.01, min_count: int=1, seed: int=42):

        self.algo = algo
        self.window = window
        self.dimensions = dimensions
        self.workers = workers
        self.epochs = epochs
        self.down_sampling = down_sampling
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.seed = seed
        self.model = None
        self.words = words
       
    def fit(self, documents, tags):
        self._set_seed()
        print ("DOC2VEC Training")
        model = Doc2Vec(documents,
                        vector_size=self.dimensions,
                        window=self.window,
                        min_count=self.min_count,
                        dm=self.algo,
                        sample=self.down_sampling,
                        workers=self.workers,
                        iter=self.epochs,
                        dbow_words = self.words,
                        alpha=self.learning_rate,
                        seed=self.seed)

        now = datetime.now()
        timestamp = datetime.timestamp(now)
        model.save("models/" + str(timestamp) + "_model_doc.dat")
        self._embedding = np.array([model.docvecs[tags[i]] for i, _ in enumerate(documents)])
        self.model = model

    def _get_embeddings(self, tags):
        embeddings = []
        for tag in tags:
            #print(tag)
            embeddings.append(self.model.docvecs[tag])      
        return(np.array(embeddings))

    def _get_single_embedding(self, tag):
        return(self.model.docvecs[tag])

    def _predict(self, vector):
        self.model.wv.similar_by_vector(vector)
    
    def get_embedding(self) -> np.array:
        embedding = self._embedding
        return embedding
