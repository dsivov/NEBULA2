from nebula_api.nebula_enrichment_api import NRE_API
from nebula_api.scene_graph_enrichment_api import SCENE_GRAPH_EXPERT_API
import time

class SceneGraphBuilder:
    def __init__(self):
        self.nre = NRE_API()
        self.scene_graph = SCENE_GRAPH_EXPERT_API() 
        print("Scene Graph Builder")
    
    def run_reconciliation(self):
        #self.nre.init_new_db("nebula_development")
        while True:
            print("Start SceneGraph plugin - waiting for new scene graph")
            movies = self.nre.wait_for_change("SceneGraph","StoryLine")
            print("run reconciliation - SceneGraph")
            #print(movies)
            for movie in movies:
                print("Processing Movie: ", movie)
                self.scene_graph.create_scene_graph(movie)
                print("Done scene graph processing for movie: ", movie )
            self.nre.update_expert_status("SceneGraph")
            time.sleep(3)
