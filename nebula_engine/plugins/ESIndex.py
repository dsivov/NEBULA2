from nebula_api.nebula_enrichment_api import NRE_API
from nebula_api.elastic_index_api import ELASTIC_SEARCH
import time

class ElasticSearchBuilder:
    def __init__(self):
        self.nre = NRE_API() 
        self.es = ELASTIC_SEARCH()
        print("Elastic Search IndexBuilder")

    
    def run_reconciliation(self):
        print("Start ElasicSearch plugin - waiting for new  ES index")
        while True:
            movies = self.nre.wait_for_change("ESIndex","StoryLine")
            print("run reconciliation - ESIndex")
            for movie in movies:
                print("Processing Movie: ", movie)
                self.es.insert_update_index(movie)
                #self.es.create_add_index(doc)
            print("Index created for: ", movies)
            self.nre.update_expert_status("ESIndex")
            time.sleep(3)
