from nebula_api.nebula_enrichment_api import NRE_API
from nebula_api.story_line_api import STORY_LINE_API
import time

class StoryLineBuilder:
    def __init__(self):
        self.nre = NRE_API()
        self.story_line = STORY_LINE_API() 
        print("Story Line Builder")
    
    def run_reconciliation(self):
        print("Start StoryLine plugin - waiting for new story line")
        while True:
            movies = self.nre.wait_for_change("StoryLine", "SceneDetector")
            print("run reconciliation - StoryLine")
            #print(movies)
            for movie in movies:
                meta_ = self.story_line.get_movie_meta(movie)
                meta = meta_[movie]
                print("Processing Movie: ", movie)
                for i, scene_element in enumerate(meta['scene_elements']):
                    file_name = meta['full_path']
                    movie_id = meta['movie_id']
                    arango_id = meta['_id']
                    mdfs = meta['mdfs'][i]
                    start_frame = scene_element[0]
                    stop_frame = scene_element[1]
                    stage = i
                    self.story_line.create_story_line(file_name, movie_id, arango_id, stage, start_frame, stop_frame, mdfs)
            self.nre.update_expert_status("StoryLine")
            time.sleep(3)
