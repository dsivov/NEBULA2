from nebula_api.nebula_enrichment_api import NRE_API
from openie import StanfordOpenIE
import spacy
from gensim.parsing.preprocessing import remove_stopwords

properties = {
    'openie.affinity_probability_cap': 2 / 3,
}

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
def get_lemma(sentence):
    # Parse the sentence using the loaded 'en' model object `nlp`
    #tokens = []
    filtered_sentence = remove_stopwords(sentence.lower())
    ts = self.nlp(filtered_sentence)
    for token in ts:
        if token.lemma_ != "-PRON-":
            tokens = ' '.split(token.lemma_)
        #" ".join([token.lemma_ for token in doc])
        #> 'the strip bat be hang on -PRON- foot for good'
    return(tokens)

def get_movie_meta(db, movie_id):
    with StanfordOpenIE(properties=properties) as client:
        nebula_movies = {}
        #print(movie_id)
        #query = 'FOR doc IN Movies FILTER doc.split == \'0\' AND doc.splits_total == \'30\' RETURN doc'
        query = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}" RETURN doc.sentences'.format(
            movie_id)
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/10715274\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/17342682\' RETURN doc'
        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/12911567\' RETURN doc'

        #query = 'FOR doc IN Movies FILTER doc._id == \'Movies/11723602\' RETURN doc'
        cursor = db.aql.execute(query)
        for data in cursor:
            for sents in data:
                text = ' '.join(sents)
                print('Text: %s.' % text)
                for triple in client.annotate(text):
                    print('|-', triple)

    

def pegasus_to_triplets():
    with StanfordOpenIE(properties=properties) as client:
        with open('/home/dimas/pegasus/pegasus_dataset/pegasus_sentences.dataset.dat', encoding='utf8') as r:
            corpus = r.read().replace('\n', ' ').replace('\r', '')

        triples_corpus = client.annotate(corpus[0:50000])
        print('Corpus: %s [...].' % corpus[0:800])
        print('Found %s triples in the corpus.' % len(triples_corpus))
        for triple in triples_corpus:
            print('|-', triple)
        print('[...]')
                 
       

nre = NRE_API()
db = nre.db
get_movie_meta(db, "Movies/97641820")
#pegasus_to_triplets()
