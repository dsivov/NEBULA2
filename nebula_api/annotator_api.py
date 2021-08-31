import cv2
from math import *
import os



class Annotator:
    # movie_id - UUID from VideoTagGrapBuilder class load_data()
    # db (get from VideoTagGrapBuilder class, connect_db())
    # video_file
    # type: all, by_type, by_actors
    # type = person, car...
    # actors = ["Actors/id", "Actors/id"]
    def __init__(self, movie_id, movie_uuid, db, video_file, type, filter="None"):
        print(video_file)
        video_annotate(movie_id, movie_uuid, db, video_file, type, filter)


def video_annotate(movie_id, movie_uuid, db, video_file, type, filter):
    cap = cv2.VideoCapture(video_file)
    mp4fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    f_begin, f_end = 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('the video ' + video_file + ' has ' + str(f_end) + ' frames')
    print('Frames Per Second (fps): ' + str(fps))
    print('total amount of framimport oses: ' + str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    print('frame size: ' + str(W) + 'x' + str(H))

    i0 = range(f_begin, f_end, fps)
    i1 = range(f_begin + fps, f_end + fps, fps)

    frame_list = []
    tmp_filename = "/tmp/" + movie_uuid + "_tmp.mp4"
    filename = "/tmp/" + movie_uuid + ".mp4"
    writer = cv2.VideoWriter(tmp_filename, mp4fourcc, fps, (W, H))
    positions = get_annotation_from_db(db, movie_id, type, filter)
    #print(positions)
    #positions = positions_[0]
    # scene_query = 'FOR doc IN ActorToAsset FILTER doc.`start_frame` <= @frame AND doc.`end_frame` >= @frame AND doc.`movie_id` == @movie_id RETURN ({ ' \
    #               'From:(RETURN DOCUMENT(doc._from).description),Rel: doc.`relation`,To:(RETURN DOCUMENT(' \
    #               'doc._to).description) })'
    scene_query = 'FOR doc IN Relations FILTER doc.`start_frame` <= @frame AND doc.`end_frame` >= @frame ' \
                  'AND doc.`movie_id` == @movie_id ' \
                  'RETURN ({belongs:doc.belongs_to, relation:doc.annotator})'
   
    # for pos in positions:
    Xc_old = H

    for i in range(f_begin, f_end):
        ret, frame = cap.read()
        if len(positions) > i:
            for data in positions[i]['data']:
                if 'left' in data['pos']['normalizedBoundingBox'] \
                        and 'top' in data['pos']['normalizedBoundingBox'] \
                        and 'bottom' in data['pos']['normalizedBoundingBox'] \
                        and 'right' in data['pos']['normalizedBoundingBox']:
                    x = float(data['pos']['normalizedBoundingBox']['left'])
                    w = float(data['pos']['normalizedBoundingBox']['top'])
                    y = float(data['pos']['normalizedBoundingBox']['bottom'])
                    h = float(data['pos']['normalizedBoundingBox']['right'])
                    if data['pos']['bbox_source'] == "google":
                        color = (255, 0, 0)
                    elif data['pos']['bbox_source'] == "scenegraph":
                        color = (255, 255, 255)
                    else:
                        continue

                    Xc, Yc = center_location(((x, w), (h, y)))
                    if Xc == Xc_old:
                        Xc_old = Xc
                        Xc = Xc - 50
                    actor = data['actor']
                    cv2.rectangle(frame, (int(x * W), int(w * H)), (int(h * W), int(y * H)), color, 2)
                    # cv2.putText(frame, actor, (int(x * W), int(w * H)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
                    cv2.putText(frame, actor, (int(Yc * W), int(Xc * H)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
            bind_vars = {'movie_id': movie_id, 'frame': i}
            scene_data = db.aql.execute(
                scene_query,
                bind_vars=bind_vars
            )
            Yrel = 40
            cv2.putText(frame, "Relations on Frame#: " + str(i),
                        (50, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                        (0, 0, 255), thickness=2)
            for scene in scene_data:
                #print(scene['From'][0], " ", scene['Rel'][0], " ", scene['To'][0], "Frame:", i)
                print(scene['relation'])
                if 'relation' in scene:
                    cv2.putText(frame, scene['relation'],
                                (50, Yrel), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                (255, 255, 255), thickness=1)
                Yrel = Yrel + 20
        frame_list.append(frame)

    for f in frame_list:
        writer.write(f)
    writer.release()
    cap.release()
    os.system('ffmpeg -loglevel panic -y -i ' + tmp_filename +
              ' -vcodec libx264 -an -crf 23 ' + filename)
    os.remove(tmp_filename)


def center_location(locations):
    x = 0
    y = 0
    z = 0

    for lat, lon in locations:
        lat = float(lat)
        lon = float(lon)
        x += cos(lat) * cos(lon)
        y += cos(lat) * sin(lon)
        z += sin(lat)

    x = float(x / len(locations))
    y = float(y / len(locations))
    z = float(z / len(locations))

    return atan2(y, x), atan2(z, sqrt(x * x + y * y))


def get_annotation_from_db(db, movie_id, type_, filter_):
    # scene_query = 'FOR c IN Frames SORT c.frame_number RETURN ({frame: c.frame_number, data: ' \
    #               '(FOR v,e  IN 1..1 INBOUND c Positions FILTER v.movie_id == \"265e3f714d0846839de34d02ad2decd9\"' \
    #               ' RETURN ({Actor: v.description,  Next: ' \
    #               '(FOR j,r IN OUTBOUND v ActorToAsset RETURN {RelPos: ' \
    #               '(FOR t,d IN  OUTBOUND SHORTEST_PATH j TO c Positions FILTER v.movie_id == \"265e3f714d0846839de34d02ad2decd9\"' \
    #               'RETURN ({FromPos: e.position.normalizedBoundingBox, ToPos: d.position.normalizedBoundingBox, ' \
    #               'relation: r.relation}))}) }))}) '
    bind_vars = {'movie_id': movie_id}
    if type_ == "all":
        query = 'FOR c IN Frames SORT c.frame_number RETURN ({frame: c.frame_number, data: ' \
                '(FOR v,e  IN 1..1 INBOUND c Positions FILTER v.movie_id == @movie_id RETURN ' \
                '{pos: e.position, actor: v.description} )})'
        bind_vars = {'movie_id': movie_id}
        positions = db.aql.execute(
            query,
            bind_vars=bind_vars
        )
    elif type_ == "by_type":
        query = 'FOR c IN Frames SORT c.frame_number RETURN ({frame: c.frame_number, data: ' \
                '(FOR v,e  IN 1..2 INBOUND c Positions FILTER (v.description == @filter and v.movie_id == @movie_id) ' \
                ' RETURN ' \
                '{pos: e.position, actor: v.description} )})'
        bind_vars = {'filter': filter_, 'movie_id': movie_id}
        positions = db.aql.execute(
            query,
            bind_vars=bind_vars
        )
    elif type_ == "by_actors":
        query = 'FOR c IN Frames SORT c.frame_number RETURN ({frame: c.frame_number, data: ' \
                '(FOR v,e  IN 1..2 INBOUND c Positions FILTER (v._id == @filter and v.movie_id == @movie_id) ' \
                ' RETURN' \
                '{pos: e.position, actor: v.description} )})'
        bind_vars = {'filter': filter_, 'movie_id': movie_id}
        positions = db.aql.execute(
            query,
            bind_vars=bind_vars
        )
    ret1 = [position for position in positions]
    
    return ret1


def main():
    from gdb.databaseconnect import DatabaseConnector as dbc
    t = dbc()
    db = t.connect_db('nebula_dev')
    Annotator("9394c08026674d5c878400b488fb6709", db,
              "/tmp/sceneclipautoautotrain00270.avi", "all")


if __name__ == '__main__':
    main()
