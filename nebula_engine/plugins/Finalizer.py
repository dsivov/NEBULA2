from nebula_api.nebula_enrichment_api import NRE_API
import time

class Finalizer:
    def __init__(self):
        self.nre = NRE_API()
    def run_reconciliation(self):
        print("Start Finalizer  - waiting for all processes finish")
        while True:
            movies = self.nre.wait_for_change("Finalizer", "ClipScene")
            print("run reconciliation - Finalizer for movies: ", movies)
            for movie in movies:
                query = 'FOR doc IN Movies FILTER doc._id == "' + movie + '" UPDATE doc WITH {status: \"updated\"} IN Movies'
                print(query)
                self.nre.db.aql.execute(query)
            self.nre.update_expert_status("Finalizer")
            time.sleep(3)
