from benchmark.clip_benchmark import NebulaVideoEvaluation
import numpy as np

def location_list():
    locations = ['wedding', 'boat', 'forest', 'parking lot', 'sea', 'river', 'sea shore', 'field', 'road', 'apartment',
                 'kitchen', 'restaraunt', 'zoo', 'stadium', 'car', 'office', 'backyard', 'hotel', 'entrance',
                 'train station', 'stairs', 'shower', 'military camp', 'summer camp', 'city', 'village', 'bridge',
                 'jungle', 'dessert', 'roof', 'rooftop', 'restaraunt', 'bar', 'bed', 'hospital', 'battle',
                 'street', 'palace', 'playground', 'kindergarten', 'school', 'church', 'plane', 'warehouse',
                 'ship', 'bathroom', 'bank', 'building', 'park', 'theater', 'doorway', 'protest', 'war', 'library',
                 'shop', 'ocean', 'party', 'medieval town', 'spa', 'bedroom', 'hospital']

    return locations

class LocationList:
    def __init__(self):
        self.locations = location_list()
        clip = NebulaVideoEvaluation()
        self.loc_emb = []
        for loc in self.locations:
            text_emb = clip.encode_text(loc)
            text_emb = text_emb / np.linalg.norm(text_emb)
            self.loc_emb.append(text_emb)
