from arango import ArangoClient
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import SimpleQueryString
import spacy
from gensim.parsing.preprocessing import remove_stopwords
import time
from nebula_api.nebula_enrichment_api import NRE_API

class ELASTIC_SEARCH:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.nre = NRE_API()
        self.db = self.nre.db
        self.es_host = self.nre.es_host
        self.index_name = self.nre.index_name
        self.es = Elasticsearch(hosts=[self.es_host])
        response = self.es.index(
            index = self.index_name, doc_type = "_doc",
            body = {}
        )
        assert response['result'] == 'created'

    def get_sentences(self, m):
        query = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}"  RETURN doc'.format(m)
        cursor = self.db.aql.execute(query)
        #print("Movie: ", m)
        all_tokens = []
        all_sentences = []
        for i, data in enumerate(cursor):
            #print("SCENE ELEM.: ", data['scene_element'])
            for sentences in data['sentences']:
                for sentence in sentences:
                    all_sentences.append(sentence)
                    #print(sentence)
                    #story_line = story_line + sentence + " "
                    tokens = self.get_lemma(sentence)
                    #print(tokens)
                    all_tokens = all_tokens + tokens
        return(all_tokens, all_sentences)

    def get_scenes(self, m):
        query = 'FOR doc IN StoryLine FILTER doc.arango_id == "{}"  RETURN doc'.format(m)
        cursor = self.db.aql.execute(query)
        #print("Movie: ", m)
        all_tokens = []
        for i, data in enumerate(cursor):
            #print("SCENE ELEM.: ", data['scene_element'])
            for sentences in data['scene_graph_triplets']:
                for sentence in sentences:
                    scene = " ".join(sentence) 
                    #story_line = story_line + scene + ". "
                    tokens = self.get_lemma(scene)
                    all_tokens = all_tokens + tokens
                    #print(tokens)
        return(all_tokens)

    def get_all_story_lines(self):
        query = 'FOR doc IN Movies RETURN doc'
        cursor = self.db.aql.execute(query)
        movies=[movie for movie in cursor]
        docs=[]
        for metadata in movies:
            movie_id=metadata['_id']
            description, story = self.get_all_tags(movie_id)
            main_tags = self.get_scenes(movie_id)
            scenes_number = len(metadata['scenes']) -1 
            start_time = time.strftime('%M:%S',time.gmtime(metadata['scenes'][0][0] / metadata['meta']['fps']))
            stop_time = time.strftime('%M:%S',time.gmtime(metadata['scenes'][scenes_number][1] / metadata['meta']['fps']))
            slice_interval = start_time + "-" + stop_time 
            url_path = "/" + metadata['url_path'].replace(".avi", ".mp4", 1) 
            #print(slice_interval)
            doc ={'movie_name':metadata['movie_name'], 'video':"/" + metadata['url_path'],
            'url': url_path, 'slice_interval': slice_interval,
            'timestamp':'0', 'description':description, 'tags':metadata['tags'], 'main_tags':main_tags,  'story': story,
            'movie_time_begin' : '0', 'movie_time_end' : '001', 'scenes': metadata['scenes'], 'scene_elements': metadata['scene_elements'],
            'confidence' : [], 'parents':'', 'db_id' : metadata['_id']}
            docs.append(doc)
        #print("DOCS: ",docs)
        return(docs)

    def get_single_story_line(self, movie_id):
        query = 'FOR doc IN Movies FILTER doc._id == "{}"  RETURN doc'.format(movie_id)
        print("DEBUG: ", query)
        cursor = self.db.aql.execute(query)
        movies=[movie for movie in cursor]
        docs=[]
        for metadata in movies:
            movie_id=metadata['_id']
            description, story = self.get_all_tags(movie_id)
            main_tags = self.get_scenes(movie_id)
            scenes_number = len(metadata['scenes']) -1 
            start_time = time.strftime('%M:%S',time.gmtime(metadata['scenes'][0][0] / metadata['meta']['fps']))
            stop_time = time.strftime('%M:%S',time.gmtime(metadata['scenes'][scenes_number][1] / metadata['meta']['fps']))
            slice_interval = start_time + "-" + stop_time  
            url_path = "/" + metadata['url_path'].replace(".avi", ".mp4", 1)
            #print(slice_interval)
            doc ={'movie_name':metadata['movie_name'], 'video':"/" + metadata['url_path'],
            'url': url_path, 'slice_interval': slice_interval,
            'timestamp':'0', 'description':description, 'tags':metadata['tags'], 'main_tags':main_tags, 'story': story,
            'movie_time_begin' : '0', 'movie_time_end' : '001', 'scenes': metadata['scenes'], 'scene_elements': metadata['scene_elements'],
            'confidence' : [], 'parents':'', 'db_id' : metadata['_id']}
            print("DEBUG: ", doc)
            docs.append(doc)
        print("DOCS: ",docs)
        return(docs[0])

    def get_lemma(self, sentence):
        # Parse the sentence using the loaded 'en' model object `nlp`
        tokens = []
        filtered_sentence = remove_stopwords(sentence.lower())
        ts = self.nlp(filtered_sentence)
        for token in ts:
            if token.lemma_ != "-PRON-":
                tokens.append(token.lemma_)
        #" ".join([token.lemma_ for token in doc])
        #> 'the strip bat be hang on -PRON- foot for good'
        return(tokens)

    def get_all_tags(self, movie_id):
        descriptions = []
        all_tokens, all_sentences = self.get_sentences(movie_id)
        descriptions = descriptions + all_tokens
        descriptions = descriptions + self.get_scenes(movie_id)
        descriptions = list(dict.fromkeys(descriptions))
        #print(descriptions)
        descriptions = " ".join(descriptions)
        #print(descriptions)
        return(descriptions, all_sentences)
    
    def create_add_index(self, doc):
       
        db_id = doc['db_id']
        #print (db_id)
        #self.es.delete_by_query(index=self.index_name, body={"query": {"match": {'doc.db_id': db_id }}})
        response = self.es.index(
            index = self.index_name, doc_type = "_doc",
            body = {'doc':doc}
        )
        assert response['result'] == 'created'
    
    def rebuild_index(self):
        self.es.indices.delete(index=self.index_name, ignore=[400, 404])
        docs = self.get_all_story_lines()
        for doc in docs:
            self.create_add_index(doc)
    
    def search_for_existing_movie(self, movie):
        query = {
            "match_phrase": {
                "doc.db_id": movie
            }
        }
        #step = settings.STEP_SEARCH_RESULTS
        #number_of_steps = settings.MAX_SEARCH_RESULTS // step
        #start = 0
        s = Search(using=self.es, index=self.index_name).query(query)
        s = s.highlight_options(order='score')
        for hit in s.execute():
            print("ID: ", hit.meta.id)
            return(hit.meta.id)
    
    def update_index(self, _id, _doc):
        print(_doc)
        _doc['movie_time_end'] = "002"
        self.es.update(index=self.index_name, id=_id, body={"doc": _doc})
    
    def insert_update_index(self, movie):
        _id = self.search_for_existing_movie(movie)
        if _id:
            print("Found existing index for ", movie, " updating")
            _doc = self.get_single_story_line(movie)
            self.update_index(_id, _doc)
        else:
            print("New index inserting")
            _doc = self.get_single_story_line(movie)
            self.create_add_index(_doc)


def main():
    
    es_load = ELASTIC_SEARCH()
    #es_load.rebuild_index()
    _id = es_load.search_for_existing_movie("Movies/92363515")
    doc = es_load.get_single_story_line("Movies/92363515")
    es_load.update_index(_id, doc)
    #es_load.create_add_index(es_load.get_single_story_line('Movies/92349435'))
        
if __name__ == "__main__":
    main()

    
    
    # docs = get_story_line(movie_id)
    # for doc in docs:
    #     print(doc)
    # print(docs)
    
