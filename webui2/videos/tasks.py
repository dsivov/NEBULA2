import boto3
from celery import shared_task
from collections import defaultdict
from decimal import Decimal as D
from datetime import datetime, timedelta
import json
from itertools import chain
from kafka import KafkaProducer
import logging
import time
from operator import itemgetter
from django.conf import settings
#from nebweb.settings import mon_api

from .models import Upload, VideoProcessingMonitor, TemporaryVideoFile


logger = logging.getLogger('nebweb')


@shared_task
def upload_video_to_s3_async(request_data, tmp_file_pk):
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    tmp_file_obj = TemporaryVideoFile.objects.get(pk=tmp_file_pk)
    video_from_filesystem = tmp_file_obj.video.file
    video_from_filesystem_name = video_from_filesystem.name = tmp_file_obj.video_name

    filename = (
            settings.MEDIA_ROOT +
            "/videos/" +
            video_from_filesystem_name
    )

    slices_folder_path = "{}/".format(filename)
    s3_resource = boto3.resource(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_S3_REGION_NAME
    )
    bucket = s3_resource.Bucket(settings.AWS_STORAGE_BUCKET_NAME)
    bucket.objects.filter(Prefix=slices_folder_path).delete()

    monitor_message = {
        'opcode': 'op_start',
        'key': video_from_filesystem_name,
        'event': 's3-upload',
        'tm': time.time_ns(),
        'phase': 'Start'
    }
    producer.send(settings.MONITOR_CHANNEL, value=monitor_message)

    vpm = VideoProcessingMonitor.objects.get(
        video_name=video_from_filesystem_name
    )
    vpm.status = VideoProcessingMonitor.StatusChoices.Uploading
    vpm.save()

    upload = Upload(file=video_from_filesystem)
    upload.save()

    vpm.refresh_from_db()
    vpm.status = VideoProcessingMonitor.StatusChoices.Uploaded
    vpm.save()

    vtag_message = {
        'opcode': 'op_slice',
        'time': int(request_data['segment']),
        'file': filename,
        #'camera': camera,
        'log': request_data['log_data']
    }
    logger.info('send msg:{} to vtag'.format(vtag_message))
    producer.send(settings.VTAG_CHANNEL, value=vtag_message)

    monitor_message = {
        'opcode': 'op_stop',
        'key': video_from_filesystem_name,
        'event': 's3-upload',
        'tm': time.time_ns(),
        'phase': 'Start'
    }
    producer.send(settings.MONITOR_CHANNEL, value=monitor_message)

    tmp_file_obj.delete()
    producer.flush()


def get_video_status_from_video_events(video_events):
    # Your code to determine video status here
    return VideoProcessingMonitor.StatusChoices.Processing


def time_ns_to_timestamp(time_ns_str):
    unix_time_seconds = int(time_ns_str) // 1e9
    #   modulo = int(time_ns_str) % 1e9
    t = datetime.utcfromtimestamp(unix_time_seconds)
    return t.strftime('%Y-%m-%d %H:%M:%S')


def parse_video_events(status_to_slices_ops, video_name):
    slices_data = {}
    has_slices = False
    for status, slice_ops in status_to_slices_ops.items():
        for slice_op_events in slice_ops:
            slice_name = slice_op_events['slice_name']
            # slice_name = <video_name>:<slice>
            # If slice_name != video_name remove <video_name>: (first occurrence) from it to display only slice name
            if slice_name != video_name:
                has_slices = True
                slice_name = slice_name.replace(
                    "{}:".format(video_name),
                    "",
                    1
                )

            slice_events_parsed_data = defaultdict(dict)
            for event_name, time_str in slice_op_events['data'].items():
                if event_name.endswith(":s"):
                    slice_event_name = event_name[:-len(":s")]
                    slice_events_parsed_data[slice_event_name]['start'] = time_ns_to_timestamp(time_str)
                elif event_name.endswith(":e"):
                    slice_event_name = event_name[:-len(":e")]
                    slice_events_parsed_data[slice_event_name]['end'] = time_ns_to_timestamp(time_str)
                else:
                    slice_event_name = event_name
                    slice_events_parsed_data[slice_event_name]['took'] = time_str

            slices_data[slice_name] = {
                'status': status,
                'events': slice_events_parsed_data
            }
    return slices_data, has_slices


@shared_task
def fetch_processing_videos_stats():
    active_jobs_data = mon_api.get_active_jobs()

    # -------------------Parsing data------------------

    # List order is important and means status priority
    possible_statuses = ['completed', 'timedout', 'pending']

    status_to_jobs = {
        status: active_jobs_data[status]
        for status in possible_statuses
    }
    status_to_jobs_names = {
        status: list(map(itemgetter('video_name'), active_jobs_data[status]))
        for status in possible_statuses
    }

    videos_in_redis = set(chain(*status_to_jobs_names.values()))

    video_name_to_events = {
        video_name: {
            status: []
            for status in possible_statuses
        } for video_name in videos_in_redis
    }
    for status, status_jobs in status_to_jobs.items():
        for element in status_jobs:
            _video_name = element['video_name']
            video_name_to_events[_video_name][status] = element['events']

    # ---------------Updating data----------------
    # Delete Processing videos that aren't really processing
    processing_statuses = [
        VideoProcessingMonitor.StatusChoices.Processing
    ]
    VideoProcessingMonitor.objects.filter(
        status__in=processing_statuses
    ).exclude(
        video_name__in=videos_in_redis,
    ).update(status=VideoProcessingMonitor.StatusChoices.Done)

    vpms = VideoProcessingMonitor.objects.filter(
        video_name__in=videos_in_redis
    )

    # Create new VPMs if they are processing but not exist in db
    if vpms.count() != len(videos_in_redis):
        vpms_to_create = []
        for video_name in videos_in_redis.difference(
            vpms.values_list('video_name', flat=True)
        ):
            all_video_events = list(chain(*video_name_to_events[video_name].values()))
            vpms_to_create.append(
                VideoProcessingMonitor(
                    video_name=video_name,
                    status=VideoProcessingMonitor.StatusChoices.Processing,
                    upload_progress_percents=D(100.0),
                    has_slices=len(all_video_events) > 1
                )
            )
        VideoProcessingMonitor.objects.bulk_create(vpms_to_create)

    # Update vpms except created ones
    for vpm in vpms:
        events = video_name_to_events[vpm.video_name]
        parsed_events, has_slices = parse_video_events(events, vpm.video_name)
        vpm.status = get_video_status_from_video_events(events)
        vpm.events = json.dumps(parsed_events)
        vpm.has_slices = has_slices or vpm.has_slices
        vpm.save()

    #VideoProcessingMonitor.objects.bulk_update(vpms, ['status', 'has_slices', 'events'])
