import string
from random import choices, random
import operator
from nebula_api.es import Esearch


def get_video(id): 
    return {
        'name': 'Baby Driver (2017) – Police Car Chase',
        'slice_name': 'Baby Driver 2017 Blues Explosion Chase Scene 110.mp4',
        'url': '/static/video/baby_driver.mp4',
        'timestamp': '',
        'db_id': id,
        'slice_interval': '20:00-23:12',
        'main_tags': ','.join(['car', '4 persons', 'gun']),
        'tags': [
            'car', '4 persons', 'gun'
        ],
        'graph_history': "",
        'metadata': 'Baby Driver (2017) - Blues Explosion Chase Scene',
        'match': "%.2f" % (random() * 0.5 + 0.5)
    }


def get_video_es(query, page_size=5, page=1):
    from_index = (page - 1) * page_size
    to_index = from_index + page_size

    count, results = Esearch().get_results_doc(query, from_index, to_index)
    
    docs = []
    for result in results: 
        main_tags = []
        scenes_elements = []
        scenes = []
        story = []

        for tag in result.main_tags:
            if tag not in main_tags:
                main_tags.append(tag)

        for scene_element in result.scene_elements:
            scenes_elements.append(list(scene_element))

        for scene in result.scenes:
            scenes.append(list(scene))

        for sentece in result.story:
            story.append(sentece)

        docs.append({
            'name': result.movie_name,
            'slice_name': result.movie_name + ".avi",
            'url': result.url,
            'timestamp': '',
            'db_id': result.db_id,
            'slice_interval': str(result.slice_interval),
            'main_tags':  ','.join(main_tags),
            'tags': list(result.tags),
            'graph_history': story,
            'scene_elements': scenes_elements,
            'scenes': scenes,
            'metadata': 'Nebula Hollywood movie',
            'match': result['score']
        })
    return count, docs


def get_video_moments(id, txt, search_engine):
    print("Video Moments: ", id , " ", txt)
    return [
        {
            'src': '/static/img/1-0135.jpeg',
            'position_to_display': "01:35",
            'position': 95
        },
        {
            'src': '/static/img/2-0022.jpg',
            'position_to_display': "00:22",
            'position': 22
        },
        {
            'src': '/static/img/3-0149.jpeg',
            'position_to_display': "01:49",
            'position': 109
        },
        {
            'src': '/static/img/4-0116.jpg',
            'position_to_display': "01:16",
            'position': 76
        },
    ]


def get_random_video_list(length):
    print("Random....")
    return list(sorted(sorted([
        get_video(
            'Graph/' + ''.join(list(choices(string.ascii_letters, k=8)))
        ) for _ in range(length)
    ], key=operator.itemgetter('name')), key=operator.itemgetter('match'), reverse=True))


def get_video_list(query, page_size, page):
    # Returns count, results
    return get_video_es(query, page_size, page)


def get_one_video(query):
    count, res = get_video_es(query)
    return res[0]


def get_video_recommendations(id, position, search_engine):
    print("Video Recommendations: ",id, " ", position)
    similar_scenes = []  
    similar_scenes_ = search_engine.get_similar_scenes(id, position, 10)
    for ss in similar_scenes_.values():
        similar_scenes.append({
            'src': "/" + ss['url_path'].replace(".avi", ".mp4", 1),
            'name': ss['movie_name'],
            'db_id': ss['_id'],
            'position_to_display': ss['scene_meta']['sentence'],
            'position': ss['scene_meta']['frame_number'] // ss['meta']['fps']
        })
    return(similar_scenes)
    

def get_video_scenes(video):
    #print("Get videos: ", video)
    fps = 28
    scenes = []
    for scene_element in video['scene_elements']:
        start = scene_element[0] // fps
        stop = scene_element[1] // fps
        scenes.append([start, stop])
    #print(scenes)
    return(scenes)
   
def get_similarity_algorithms_available():
   
    return [('clip2bert', 'CLIP2BERT'), ('string2clip','STRING2CLIP'), ('doc2vec', 'GRAPH2VEC')]

