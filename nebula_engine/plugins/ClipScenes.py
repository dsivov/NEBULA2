from time import time
from nebula_api.nebula_enrichment_api import NRE_API
from nebula_api.clip_scenes_enrichment_api import STORE_SCENE
import time

class ClipScenesBuilder:
    def __init__(self):
        self.nre = NRE_API()
        self.story_scene = STORE_SCENE() 
        print("Clip Scene Builder")
    
    def run_reconciliation(self):
        while True:
            print("Start ClipScene plugin - waiting for new clip scene")
            movies = self.nre.wait_for_change("ClipScene","StoryLine")
            print("run reconciliation - ClipScene")
            #print(movies)
            for movie in movies:
                print("Processing Movie: ", movie)
                self.story_scene.delete_all_vectors(movie)
                self.story_scene.create_clip_scene(movie)
                print("Clip Scene cretaed")
                self.story_scene.encode_movie_to_bert(movie)
                #encode_movie_to_bert()
            self.nre.update_expert_status("ClipScene")
            time.sleep(3)
