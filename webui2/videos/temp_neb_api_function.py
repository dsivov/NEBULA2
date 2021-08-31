import string
from random import choices, random
import operator


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


def get_video_moments(id, txt):
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
    return list(sorted(sorted([
        get_video(
            'Graph/' + ''.join(list(choices(string.ascii_letters, k=8)))
        ) for _ in range(length)
    ], key=operator.itemgetter('name')), key=operator.itemgetter('match'), reverse=True))


def get_video_recommendations(id, position):
    return [
        {
            'src': '/static/video/matrix.mp4',
            'name': 'The Matrix Reloaded (2003)',
            'db_id': 'Graph/' + ''.join(list(choices(string.ascii_letters, k=8))),
            'position_to_display': '00:00-02:00',
            'position': 0
        },
        {
            'src': '/static/video/streak.mp4',
            'name': 'Blue Streak (1999)',
            'db_id': 'Graph/' + ''.join(list(choices(string.ascii_letters, k=8))),
            'position_to_display': '01:21-03:07',
            'position': 81
        },
        {
            'src': '/static/video/bourne.mp4',
            'name': 'The Bourne Ultimatum (2007)',
            'db_id': 'Graph/' + ''.join(list(choices(string.ascii_letters, k=8))),
            'position_to_display': '00:10-01:55',
            'position': 10
        },
        {
            'src': '/static/video/dukes.mp4',
            'name': 'The Dukes of Hazzard (2005)',
            'db_id': 'Graph/' + ''.join(list(choices(string.ascii_letters, k=8))),
            'position_to_display': '00:20-02:45',
            'position': 20
        },
        {
            'src': '/static/video/quantum.mp4',
            'name': 'Quantum of Solace (2008)',
            'db_id': 'Graph/' + ''.join(list(choices(string.ascii_letters, k=8))),
            'position_to_display': '00:03-00:40',
            'position': 3
        },
        {
            'src': '/static/video/barnyard.mp4',
            'name': 'Barnyard - A Cow In Our Car! (2006)',
            'db_id': 'Graph/' + ''.join(list(choices(string.ascii_letters, k=8))),
            'position_to_display': '01:00-01:23',
            'position': 60
        }
    ]


def get_video_scenes(video):
    return [(0, 7), (7, 54), (54, 73), (73, 113), (113, 148), (148, 165), (165, 192)]


def get_similarity_algorithms_available():
    return [('gd2v', 'Graph -> Doc2Vec'), ('bert', 'BERT')]
