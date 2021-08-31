from django.contrib import admin

# Register your models here.
from .models import Video, VideoProcessingMonitor

admin.site.register(Video)
admin.site.register(VideoProcessingMonitor)
