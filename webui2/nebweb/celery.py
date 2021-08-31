import os
from celery import Celery

from .settings import Cfg


# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'nebweb.settings')

redis_cfg = Cfg(['redis_db'])
redis_host = redis_cfg.get('redis_db', 'host')
redis_port = redis_cfg.get('redis_db', 'port')
celery_broker = "redis://{}:{}/0".format(redis_host, redis_port)

app = Celery('nebweb', broker=celery_broker)

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

app.conf.beat_schedule = {
    # Executes every 5 seconds
    'fetch-processing-videos-stats': {
        'task': 'videos.tasks.fetch_processing_videos_stats',
        'schedule': 5.0,
        'args': (),
    },
}


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')