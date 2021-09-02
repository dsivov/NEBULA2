from operator import index
import os
import json
from django.conf import settings
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import SimpleQueryString
import logging

logger = logging.getLogger('nebweb')


class Esearch:
    def __init__(self):
        config_file = settings.BASE_DIR + '/../config/config.json'

        jf = open(config_file)
        cfg = json.load(jf)['esdb']
        jf.close()
        self.es = Elasticsearch(hosts=[cfg.get('es_host')]) #, verify_certs=False)
        self.index = cfg.get('se_indx')
        print("Elastic Search Host:", cfg.get('es_host'))
        assert self.es.ping()

    def get_results(self, qin, thr=0.01):
        results = dict()
        #step = settings.STEP_SEARCH_RESULTS
        #number_of_steps = settings.MAX_SEARCH_RESULTS // step
        #start = 0
        s = Search(using=self.es).query(
            SimpleQueryString(
                query=qin,
                fields=["doc.description", "doc.movie_name", "doc.db_id"],
                default_operator='and'
            )
        ).extra(min_score=thr)

        for hit in s.scan():
            if hit.doc.db_id not in results:
                doc = hit.doc
                slice_interval = (doc.movie_time_begin, doc.movie_time_end)
                results[doc.db_id] = (
                    doc.db_id,
                    doc.movie_name,
                    doc.video,
                    doc.timestamp,
                    doc.description,
                    doc.parents,
                    slice_interval
                )

        return results
    
    def get_results_doc(self, qin, from_index=0, to_index=10, thr=0.01):
        s = Search(using=self.es, index=self.index).query(
            SimpleQueryString(
                query=qin,
                fields=["doc.description", "doc.movie_name", "doc.db_id"],
                default_operator='and'
            )
        ).extra(min_score=thr)

        count = s.count()

        s = s.highlight_options(order='score')[from_index: to_index]

        results = []
        for hit in s.execute():
            if hit.doc.db_id not in results:
                doc = hit.doc
                doc['score'] = hit.meta.score
                results.append(doc)

        return count, results
