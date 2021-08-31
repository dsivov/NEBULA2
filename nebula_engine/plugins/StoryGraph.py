from nebula_api.nebula_enrichment_api import NRE_API
from nebula_api.story_graph_enrichment_api import STORY_GRAPH_EXPERT_API
import time

class StoryGraphBuilder:
    def __init__(self):
        self.nre = NRE_API()
        self.story_graph = STORY_GRAPH_EXPERT_API() 
        print("Story Graph Builder")
    
    def run_reconciliation(self):
        while True:
            print("Start StoryGraph plugin - waiting for new story graph")
            movies = self.nre.wait_for_change("StoryGraph", "SceneGraph")
            print("run reconciliation - StoryGraph")
            #print(movies)
            for movie in movies:
                self.story_graph.create_story_graph(movie)
                print("Processing Movie: ", movie)
            self.nre.update_expert_status("StoryGraph")
            time.sleep(3)
