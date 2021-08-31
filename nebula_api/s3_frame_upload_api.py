from nebula_enrichment_api import NRE_API
import boto3
import os
import cv2
from arango import ArangoClient


def insert_node_to_scenegraph(self, movie_id, arango_id, _class, scene_element, description, start, stop):
    query = 'UPSERT { movie_id: @movie_id, description: @description, scene_element: @scene_element} INSERT  \
            { movie_id: @movie_id, arango_id: @arango_id, class: "Object", description: @class, scores: @scores, bboxes: @bboxes \
                 scene_element: @scene_element, start: @start, stop: @stop, step: 1} UPDATE \
                { step: OLD.step + 1 } IN Nodes \
                    RETURN { doc: NEW, type: OLD ? \'update\' : \'insert\' }'
    bind_vars = {'movie_id': movie_id,
                    'arango_id': arango_id,
                    'class': _class,
                    'scene_element': scene_element,
                    'start': start,
                    'stop': stop,
                    'description': description
                   }
    self.db.aql.execute(query, bind_vars=bind_vars)
    

def divide_movie_into_frames(movie_in_path, movie_out_folder):
    cap = cv2.VideoCapture(movie_in_path)
    ret, frame = cap.read()
    num = 0
    cv2.imwrite(os.path.join(movie_out_folder,
                                f'frame{num:04}.jpg'), frame)
    while cap.isOpened() and ret:
        num = num + 1
        ret, frame = cap.read()
        if frame is not None:
            cv2.imwrite(os.path.join(movie_out_folder,
                                        f'frame{num:04}.jpg'), frame)
    return num

def store_frames_to_db(s3, movie_id, frames_folder, video_file):
    bucket_name = "nebula-frames"
    folder_name = movie_id
    s3.put_object(Bucket=bucket_name, Key=(folder_name+'/'))
    print(frames_folder)
    if not os.path.exists(frames_folder):
        os.mkdir(frames_folder)
    else:
        for f in os.listdir(frames_folder):
            if os.path.isfile(os.path.join(frames_folder, f)):
                os.remove(os.path.join(frames_folder, f))
    num_frames = divide_movie_into_frames(video_file, frames_folder)
    # SAVE TO REDIS - TBD
    if num_frames > 0:
        for k in range(num_frames):
            img_name = os.path.join(
                frames_folder, f'frame{k:04}.jpg')
            s3.upload_file(img_name, bucket_name, folder_name+'/' + f'frame{k:04}.jpg')




def main():
    client = ArangoClient(
        hosts='http://ec2-18-158-123-0.eu-central-1.compute.amazonaws.com:8529')
    db = client.db("nebula_development", username='nebula', password='nebula')
    #movie_id = 'Movies/92363482'
    nre = NRE_API()
    mv = []
    s3 = boto3.client('s3', region_name='eu-central-1')
    query = 'FOR doc IN Movies RETURN doc'
    cursor = db.aql.execute(query)
    for p in cursor:
        full_path = p['full_path']
        movie_id = p['_id']
        mv.append((movie_id, full_path))
    for m in mv:
        print(m[1], " ", m[0])
        store_frames_to_db(s3, m[0], "/dataset/frames", m[1])
    
    #s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=location)
    #bucket_name = "nebula-frames"
    #folder_name = movie_id
    #location = {'LocationConstraint': 'eu-central-1'}
    # movies = nre.get_all_movies()
    #for movie in movies:
    #    print("Processing Movie: ", movie)
    #   story_graph.create_story_graph(movie)

    #clip.test_clip_vectors()
    #clip.get_sentences()
if __name__ == "__main__":
    main()
