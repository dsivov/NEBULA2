from decimal import Decimal as D
import threading

from django.contrib.auth.models import User
from django.db import models
from djchoices import DjangoChoices, ChoiceItem
# from .validators import validate_video
# from decimal import Decimal
from .storage import PublicMediaStorage, OverwriteStorage


class Video(models.Model):

###    name = models.CharField(max_length=100)
###    videofile = models.FileField(upload_to='media/videos/')
#                                 storage=OverwriteStorage(),
#                                 validators=[validate_video],
#                                 verbose_name="blank",
#                                 max_length=50)
# storage=OverwriteStorage())

###    slice = models.DecimalField(max_digits=3,
###                                decimal_places=0,
###                                default=Decimal('5'))

    def __str__(self):
        return self.name + ":" + str(self.videofile)


class VideoProcessingMonitor(models.Model):
    class StatusChoices(DjangoChoices):
        Initial = ChoiceItem('initial', 'Start')
        Uploading = ChoiceItem('uploading', 'Uploading')
        Uploaded = ChoiceItem('uploaded', 'Uploaded')
        Processing = ChoiceItem('processing', 'Processing')
        Done = ChoiceItem('done', 'Done')

    created_at = models.DateTimeField(auto_now_add=True)
    video_name = models.TextField(unique=True)
    status = models.CharField(max_length=10, choices=StatusChoices.choices, default=StatusChoices.Initial)
    events = models.TextField(default="")
    upload_progress_percents = models.DecimalField(max_digits=4, decimal_places=1, default=D(0.0))
    has_slices = models.BooleanField(default=False)


class ProgressPercentage(object):

    def __init__(self, filename, file_size):
        self._filename = filename
        self._size = file_size
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            VideoProcessingMonitor.objects.filter(
                video_name=self._filename
            ).update(
                upload_progress_percents=percentage
            )


class Upload(models.Model):
    upload_at = models.DateTimeField(auto_now_add=True)
    file = models.FileField(
        upload_to='videos/',
        storage=PublicMediaStorage(
            progress_percentage_class=ProgressPercentage
        )
    )


class TemporaryVideoFile(models.Model):
    video = models.FileField(upload_to='temp/videos/', storage=OverwriteStorage())
    video_name = models.CharField(max_length=1000)


class UserSetting(models.Model):
    # TODO create user's model. we don't have any user now
    user = models.CharField(max_length=100)
    settings = models.CharField(max_length=25, null=True, blank=True)  # TODO should be foreign key to model
