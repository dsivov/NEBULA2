from numpy import tile
from nebula_api.nebula_enrichment_api import NRE_API
from nebula_api.scene_detector_api import NEBULA_SCENE_DETECTOR
import time

class SceneDetector:
    def __init__(self):
        self.nre = NRE_API()
        self.scene_detector = NEBULA_SCENE_DETECTOR() 
        print("Scene Detector")
    
    def run_reconciliation(self):
        while True:
            print("Start SceneDetector plugin - waiting for new scene")
            movies = self.nre.wait_for_change("SceneDetector","movies")
            print("run reconciliation - SceneDetector")

            #print(movies)
            #for movie in movies:
            movies= self.scene_detector.new_movies_batch_processing("/dataset/upload", "/dataset/development", "development")
            print(movies)
            for movie in movies:
                self.scene_detector.store_frames_to_s3(movie[1], "/dataset/frames/", movie[0])
            print("Done new movie processing - SceneDetector")
            self.nre.update_expert_status("SceneDetector")
            time.sleep(3)
