import math
import ast
import boto3
from botocore.exceptions import ClientError
from botocore.client import Config
import csv
from io import StringIO
from datetime import datetime
import json
from kafka import KafkaProducer
import logging
from operator import itemgetter
import os
from pathlib import Path
import random
import re
import time
import uuid
from random import choices
import string

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.views import View
from django.urls import reverse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet

from nebula_api.annotator_api import Annotator
from nebula_api.nebula_embeddings_api import EmbeddingsLoader
from nebula_api.get_graph_api import get_graph_as_dict, get_many_movies_graph_as_dict, graph_dict_to_ml_file_data
from .es import Esearch
from .forms import VideoForm, VideoSearchForm, VideoAnnotateForm, SettingsChoiceForm
from .models import Upload, VideoProcessingMonitor, UserSetting
from .serializers import (
    VideoProcessingMonitorSerializer,
    TemporaryVideoFileSerializer,
)
<<<<<<< HEAD
from .tasks import upload_video_to_s3_async, upload_video_to_server_async
=======
from .tasks import upload_video_to_s3_async
>>>>>>> 205dcbef875be86c0ab25769dffa7df9cff6dc91
from nebula_api.search_api import (
    get_video, get_video_recommendations,
    get_video_moments, get_video_scenes, get_video_list, get_one_video
)
# import Video


logger = logging.getLogger('nebweb')
try:
    embeddings_loader = EmbeddingsLoader('clip2bert')
    scenes_embeddings_loader = EmbeddingsLoader('clip2scene')
    strings_embeddings_loader = EmbeddingsLoader('clip4string')
    doc_embeddins_loader = EmbeddingsLoader('doc2vec')
except FileNotFoundError:
    embeddings_loader = None


def home_page(request):
    if request.method == "GET":
        return redirect(reverse('video-search'))
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)


def upload_file():
    print()


def video_upload_view(request):
    if request.method == 'POST':
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'))

        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        print(name)
        filename = settings.MEDIA_ROOT + "/" + name
        msg = json.dumps({
            'opcode': 'op_slice',
            'time': 5,
            'file': filename
        })
        print(msg)
        producer.send('vtag', value=msg)

    #    context = {
    #        'videofile': videofile,
    #        'form': form
    #    }

    #    return render(request, 'videos.html', context)
    return render(request, 'upload.html')


def create_presigned_url(bucket_name, object_name, expiration=3600):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client(
        's3',
        region_name='us-east-2',
        #        config=Config(s3={'addressing_style': 'path'})
    )
    try:
        response = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': object_name
            },
            HttpMethod="GET",
            ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL
    return response


s3_video_filename_pattern = re.compile(r'^media/videos/(?P<filename>[^/]+\.mp4)$')
s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket(settings.AWS_STORAGE_BUCKET_NAME)


class VideoUploadView(View):
    @staticmethod
    def list_s3_videos():
        r = []
        for o in bucket.objects.all():
            pattern_found = re.search(s3_video_filename_pattern, o.key)
            if pattern_found is not None:
                r.append(pattern_found.group('filename'))
        return r

    @staticmethod
    def _create_new_vpm_for_video(request, video_name, form, status_to_set=None):
        if VideoProcessingMonitor.objects.filter(
                video_name=video_name,
        ).exclude(
            status=VideoProcessingMonitor.StatusChoices.Done
        ).exists():
            form.add_error(None, 'Video is already processing')
            return render(request, 'videos.html', {'form': form, 'vpm': None}), None

        # Delete Done VPM for this file
        VideoProcessingMonitor.objects.filter(video_name=video_name).delete()
        # Create a new VPM
        new_vpm_data = {
            'video_name': video_name
        }
        if status_to_set is not None:
            new_vpm_data['status'] = status_to_set
        return None, VideoProcessingMonitor.objects.create(
            **new_vpm_data
        )

    @staticmethod
    def _create_temporary_file(file):
        """
        (temporary_file) -> TemporaryVideoFile.pk
        """
        s = TemporaryVideoFileSerializer(data={'video': file, 'video_name': file.name})
        s.is_valid(raise_exception=True)
        tvf = s.save()
        return tvf.pk

    def get(self, request):
        print("GET")
        available_filenames_list = self.list_s3_videos()
        form = VideoForm(
            available_filenames_list
        )
        context = {
            'form': form,
            'vpm': None
        }
        return render(request, 'videos.html', context)

    def post(self, request):
        json_available_filenames_list = request.POST.get('available_filenames_list', '')
        if json_available_filenames_list:
            available_filenames_list = json.loads(
                json_available_filenames_list
            ) or []
        else:
            available_filenames_list = self.list_s3_videos()

        form = VideoForm(
            available_filenames_list,
            request.POST,
            request.FILES
        )
        valid = form.is_valid()
        if not valid:
            return render(request, 'videos.html', {'form': form, 'vpm': None})

        #camera = ast.literal_eval(request.POST['camera_type'])

        s3_videos = request.POST.getlist('choose_video_from_s3')
        video_from_filesystem = request.FILES.get('video', '')
        uploading_file_vpm = None

        if video_from_filesystem:
            response_with_error, new_vpm = self._create_new_vpm_for_video(
                request,
                video_from_filesystem.name,
                form
            )
            if response_with_error:
                return response_with_error
            uploading_file_vpm = new_vpm

            tmp_file_pk = self._create_temporary_file(video_from_filesystem)
            upload_video_to_s3_async.delay(
                request.POST,
                tmp_file_pk
            )
        elif s3_videos:
            for s3_video in s3_videos:
                producer = KafkaProducer(
                    bootstrap_servers=['localhost:9092'],
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )

                response_with_error, new_vpm = self._create_new_vpm_for_video(
                    request,
                    s3_video,
                    form,
                    VideoProcessingMonitor.StatusChoices.Uploaded
                )
                if response_with_error:
                    return response_with_error
                filename = (
                        settings.MEDIA_ROOT +
                        "/videos/" +
                        s3_video
                )
                # Delete folder with existing slices
                slices_folder_path = "{}/".format(filename)
                bucket.objects.filter(Prefix=slices_folder_path).delete()
                # Send messages
                monitor_message = {
                    'opcode': 'op_start',
                    'key': s3_video,
                    'event': 's3-upload',
                    'tm': time.time_ns(),
                    'phase': 'Start'
                }
                producer.send(settings.MONITOR_CHANNEL, value=monitor_message)

                vtag_message = {
                    'opcode': 'op_slice',
                    'time': int(request.POST['segment']),
                    'file': filename,
                    #'camera': camera,
                    'log': request.POST['log_data']
                }
                producer.send(settings.VTAG_CHANNEL, value=vtag_message)
                logger.info('send msg:{} to vtag'.format(vtag_message))

                monitor_message = {
                    'opcode': 'op_stop',
                    'key': s3_video,
                    'event': 's3-upload',
                    'tm': time.time_ns(),
                    'phase': 'Stop'
                }
                producer.send(settings.MONITOR_CHANNEL, value=monitor_message)

                producer.flush()
        return render(
            request,
            'videos.html',
            {
                'form': form,
                'vpm': uploading_file_vpm
            }
        )


class VideoProcessingMonitorViewSet(ModelViewSet):
    serializer_class = VideoProcessingMonitorSerializer

    def get_queryset(self):
        q = VideoProcessingMonitor.objects.all()
        if 'with_done' not in self.request.query_params:
            return q.exclude(
                status=VideoProcessingMonitor.StatusChoices.Done
            )
        return q


class VideoSearchView(View):
    @staticmethod
    def get_time_str_from_seconds(seconds_str):
        seconds = int(seconds_str)
        mins = seconds // 60
        seconds = seconds % 60
        return "%s:%s" % (str(mins).zfill(2), str(seconds).zfill(2))

    @classmethod
    def _parse_results(cls, results):
        r = []
        for db_id, movie_name, path, val, tags, main_tags, slice_interval in results.values():
            # need to cut bucket name from key
            p = Path(path)
            url = create_presigned_url(settings.AWS_STORAGE_BUCKET_NAME, str(Path(*p.parts[2:])))
            full_name = os.path.join(*p.parts[2:7])
            sp = full_name.split('splits')
            video_name = sp[0][:-1].split('/')[-1]
            slice_name = sp[1][1:]

            slice_seconds_begin, slice_seconds_end = slice_interval
            slice_interval_str = "{}-{}".format(
                cls.get_time_str_from_seconds(slice_seconds_begin),
                cls.get_time_str_from_seconds(slice_seconds_end),
            )

            r.append(
                {
                    'name': video_name,
                    'slice_name': slice_name,
                    'url': url,
                    'timestamp': val,
                    'tags': tags,
                    'main_tags': main_tags,
                    'db_id': db_id,
                    'slice_interval': slice_interval_str
                }
            )
        return r

    @staticmethod
    def _sort_results(results, sort_params):
        sort_reversed = (sort_params['sort_direction'] == 'desc')
        if sort_params['sort_by'] == 'name':
            results.sort(key=itemgetter('name'), reverse=sort_reversed)
        elif sort_params['sort_by'] == 'timestamp':
            # Replace with results.sort(lambda x: datetime.strptime(<format>), reverse=sort_reversed)
            # when you know timestamp format
            results.sort(key=lambda x: int(x['timestamp']), reverse=sort_reversed)
        return results

    def get(self, request):
        form = VideoSearchForm(request.GET or None)

        searchvalue = ''
        search_method = ''
        page = 1
        size = 100
        sort_by = ''
        sort_direction = ''
        similar_id = ''
        similarity_algo = ''

        searchresults = []
        number_of_pages = 1
        number_of_results = 0

        es = Esearch()

        if form.is_valid():
            data = form.cleaned_data
            searchvalue = data['search'] or searchvalue
            sort_by = data['sort_by'] or ''
            sort_direction = data['sort_direction'] or ''
            size = data['size'] or size
            page = data['page'] or page
            similar_id = data['SimilarID'] or similar_id
            search_method = data['search_method'] or search_method
            similarity_algo = data['similarity_algo'] or similarity_algo
            max_results = 500
            #print("size: ", size, " ", page)
        if similar_id:
            #print("ALGO: ", similarity_algo)
            #print("Search method: ", search_method)
            try:
                if similarity_algo == "clip2bert":
                    movies = embeddings_loader.get_similar_movies(
                        similar_id, size) or []
                    #print("VIEWS: ", movies)
                if similarity_algo == "string2clip":
                    movies = strings_embeddings_loader.get_similar_movies(
                        similar_id, size) or []
                if similarity_algo == "doc2vec":
                    movies = doc_embeddins_loader.get_similar_movies(
                        similar_id, size) or []
            except:
                movies = []
            results = []         
            for movie in movies:
                #print("Movie: ", movie)
                result = get_one_video(movie)
                #print(movies[movie])
                result['match'] = movies[movie]['distance']
                results.append(result)
            #results = [main_video, *get_random_video_list(50)]
            number_of_results = len(results)
            number_of_pages = math.ceil(number_of_results / size)
            if page > number_of_pages:
                form = VideoSearchForm({
                    'page': number_of_pages,
                    'size': size,
                    'search': searchvalue,
                    'search_method': search_method,
                    'similar_id': similar_id
                })
                page = number_of_pages
            start, end = (page - 1) * size, page * size
            # parsed_results = self._parse_results(results)
            searchresults = results[start: end]

        elif searchvalue:
            results = get_video_list(size, searchvalue)
            number_of_results = len(results)
            number_of_pages = math.ceil(number_of_results / size)
            if page > number_of_pages:
                form = VideoSearchForm(
                    {
                        'page': number_of_pages,
                        'size': size,
                        'search': searchvalue,
                        'search_method': search_method
                    }
                )
                page = number_of_pages
            start, end = (page - 1) * size, page * size
            #parsed_results = self._parse_results(results)
            searchresults = self._sort_results(
                    results,#parsed_results,
                {'sort_by': sort_by, 'sort_direction': sort_direction}
            )[start: end]

        number_of_pages = math.ceil(number_of_results / size)
        context = {
            'form': form,
            'searchresults': searchresults,
            'searchresults_json': json.dumps(searchresults),
            'search': searchvalue,
            'size': size,
            'search_method': search_method,
            'is_paginated': number_of_results > size,
            'number_of_results': number_of_results,
            'has_previous': page > 1,
            'has_next': page < number_of_pages,
            'previous': page - 1,
            'next': page + 1,
            'current_page': page,
            'number_of_pages': number_of_pages,
            'sort_by': sort_by,
            'sort_direction': sort_direction,
            'similar_id': similar_id
        }

        return render(request, 'video_search_test.html', context)


class ShowSimilarView(APIView):
    @classmethod
    def _parse_results(cls, results):
        r = []
        for el in results.values():
            logger.debug(el.keys())
            _key, db_id, _rev, movie_name, path, movie_id, meta, file_name, split, splits_total, status, distance = el.values()
            # need to cut bucket name from key
            p = Path(path)
            url = create_presigned_url(settings.AWS_STORAGE_BUCKET_NAME, str(Path(*p.parts[2:])))
            sp = os.path.join(*p.parts[2:7]).split('splits')
            video_name = sp[0][:-1].split('/')[-1]
            slice_name = sp[1][1:]

            r.append(
                {
                    'name': video_name,
                    'slice_name': slice_name,
                    'url': url,
                    'timestamp': '-',
                    'tags': '-',
                    'main_tags': '-',
                    'db_id': db_id,
                    'slice_interval': '-',
                    'distance': distance
                }
            )
        return r

    def get(self, request):
        db_id = request.query_params.get('id', None)
        if db_id is None:
            return Response('id is not specified', status=400)
        try:
            sims = embeddings_loader.get_similar_movies(db_id, 10) or []
        except:
            sims = []
        return Response(
            data=self._parse_results(sims),
            status=200
        )


def video_annotate(request):
    import urllib.request
    from nebula_api.cfg import Cfg
    form = VideoAnnotateForm(request.POST or None)

    video_url = form.data.get("video_url")
    db_id = form.data.get("db_id")
    local_annotate_video = os.environ.get('VIRTUAL_ENV') + '/annotate/annotate.mp4'
    urllib.request.urlretrieve(video_url, local_annotate_video)
    from nebula_api.databaseconnect import DatabaseConnector as dbc
    t = dbc()
    db = t.connect_db(settings.DB_NAME)
    movie_coll = db.collection('Movies')
    _movie_id = movie_coll.find(filters={'_id': db_id}).next()
    movie_id = _movie_id['movie_id']
    annotated_video = local_annotate_video
    filename = \
        settings.MEDIA_ROOT + \
        "/videos/annotated/" + movie_id + ".mp4"
    b3_session = boto3.Session()
    storage_client = b3_session.client('s3')
    _uuid = uuid.uuid4()
    movie_uuid = str(_uuid.hex)
    dest_blob_name = 'media/videos/annotate/' + os.path.basename(movie_uuid + ".mp4")
    cfg = Cfg(['esdb'])
    bucket_name = cfg.get('esdb', 'upload_bucket_name')
    blob = storage_client
    # Annotator("058579ff330c473f83fd070169ccf6c7", db, "17.mp4", "by_type", "car")
    
    Annotator(movie_id, movie_uuid, db, annotated_video, "all")
    blob.upload_file("/tmp/" + movie_uuid + ".mp4", bucket_name, dest_blob_name)
    os.remove("/tmp/" + movie_uuid + ".mp4")
    url = create_presigned_url(settings.AWS_STORAGE_BUCKET_NAME,
                               str(dest_blob_name))

    ###    lastvideo = Video.objects.last()
    #    import pdb; pdb.set_trace()

    context = {
        'form': form,
        'annotated_url': url,
        'db_id': db_id
    }

    return render(request, 'video_annotate.html', context)


def processing_page(request):
    if request.method == "GET":
        return render(request, 'processing.html')
    else:
        return Response("Method not allowed", status=status.HTTP_405_METHOD_NOT_ALLOWED)


def settings_page(request):
    # TODO create code/text choices. we use test code/text
    user_choice = ''
    context = {}
    user = 'Kirim'
    form = SettingsChoiceForm(request.GET or None)
    if form.is_valid():
        user_choice = request.GET.get('user_settings', '')

    if request.method == "POST":
        user_choice = request.POST.get("user_settings")
        UserSetting.objects.update_or_create(user=user, defaults={'settings': user_choice})
        value_of_setting = dict(SettingsChoiceForm().fields.get('user_settings').choices)[user_choice]
        context.update({'set_user_setting': value_of_setting})

    form = SettingsChoiceForm({'user_settings': user_choice})
    context.update({'form': form})
    return render(request, 'settings.html', context)


class VideosExportView(View):
    @staticmethod
    def _parse_data_for_csv(videos_data):
        return [
            {
                'Name': data['name'],
                'Slice Name': data['slice_name'],
                'Slice Section': data['slice_interval'],
                'Graph DB': data['db_id'],
                #'Timestamp': data['timestamp'],
                #'Main Tags': data['main_tags'],
                'Tags': data['tags'],
                'Video Link in S3': data['url'],
            } for data in videos_data
        ]

    def post(self, request):
        videos_data_json = request.POST.get('selected_videos', None)
        if not videos_data_json:
            return render(request, 'error.html', {'error': 'Videos to export data not specified'})

        videos_data = json.loads(videos_data_json)

        buff = StringIO()
        writer = csv.DictWriter(
            buff,
            fieldnames=['Name', 'Slice Section', 'Slice Name', 'Graph DB', 'Tags', 'Video Link in S3']
        )
        writer.writeheader()

        for video_data in self._parse_data_for_csv(videos_data):
            writer.writerow(video_data)

        response = HttpResponse(buff.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="ChosenVideosExport.csv"'

        return response


class ExportVideosGraphAsMLFile(APIView):
    def post(self, request):
        videos_data_json = request.POST.get('selected_videos', None)
        if not videos_data_json:
            return render(request, 'error.html', {'error': 'Videos to export data not specified'})

        videos_data = json.loads(videos_data_json)

        db_ids = list(map(itemgetter('db_id'), videos_data))

        graph_dict = get_many_movies_graph_as_dict(db_ids)
        ml_file_bytes = graph_dict_to_ml_file_data(graph_dict)

        response = HttpResponse(
            ml_file_bytes,
            content_type='application/gml+xml'
        )
        response['Content-Disposition'] = 'attachment; filename="ChosenVideosGraphDB.gml"'
        return response


class ExportVideoGraphAsJSONView(APIView):
    def get(self, request):
        db_id = request.GET.get('db_id', None)
        if db_id is None:
            return render(request, 'error.html', {'error': 'Video to export DB_ID not specified'})

        graph_dict = get_graph_as_dict(db_id)

        response = HttpResponse(
            json.dumps(graph_dict, sort_keys=True, indent=4),
            content_type='application/json'
        )
        response['Content-Disposition'] = 'attachment; filename="ChosenVideosExport.json"'
        return response


class ExportVideoGraphAsSVGView(View):
    @staticmethod
    def _parse_dict(graph_dict):
        return {
            'nodes': [
                {
                    'id': data['_id'],
                    'label': data['description'],
                    'x': random.random(),
                    'y': random.random(),
                    'size': random.random(),
                    'color': '#666'
                }
                for data in graph_dict['Nodes'].values()
            ],
            'edges': [
                {
                    'id': data['_id'],
                    'source': data['_from'],
                    'target': data['_to'],
                    'size': random.random(),
                    'color': '#ccc'
                }
                for data in graph_dict['Edges'].values()
            ]
        }

    def get(self, request):
        db_id = request.GET.get('db_id', None)
        if db_id is None:
            return render(request, 'error.html', {'error': 'Video to export DB_ID not specified'})

        graph_dict = get_graph_as_dict(db_id)

        return render(request, 'video_export_graph_svg.html', {'graph': self._parse_dict(graph_dict)})


class VideoDetailPageView(View):
    def get(self, request):
        id = request.GET['id']
        position = request.GET['position'] if 'position' in request.GET else None
       
        if position:
            position = int(position)
        else:
            position = 0
    
        video = get_one_video(id)
        video['main_tags'] = video['graph_history'][0]
        video['graph_history'] = video['graph_history'][0] + "->" + video['graph_history'][4] + "->" + video['graph_history'][8] + "..."
        scenes = get_video_scenes(video)
        context = {
            'video': video,
            'video_json': json.dumps(video),
            'id': id,
            'position': position,
            'scenes': json.dumps(scenes)
        }
       
        ren = render(request, 'video_page.html', context)
       
        return(ren)


class VideoDetailMomentsView(APIView):
    def get(self, request, *args, **kwargs):
        print("VideoDetailMomentsView ")
        if 'id' not in request.query_params:
            return Response('id is not specified', status=400)
        if 'txt' not in request.query_params:
            return Response('txt is not specified', status=400)
        print("VideoDetailMomentsView ")
        id = request.query_params['id']
        txt = request.query_params['txt']
        print("VideoDetailMomentsView ", id)
        if not txt:
            results = []
        else:
            results = get_video_moments(id, txt, scenes_embeddings_loader)

        return Response(results)


class VideoDetailRecommendationsView(APIView):
    def get(self, request, *args, **kwargs):
        
        if 'id' not in request.query_params:
            return Response('id is not specified', status=400)
        if 'position' not in request.query_params:
            return Response('position is not specified', status=400)
        id = request.query_params['id']
        position = request.query_params['position']
        results = get_video_recommendations(id, position, scenes_embeddings_loader)

        return Response(results)
