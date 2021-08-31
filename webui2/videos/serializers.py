from rest_framework import serializers

from .models import VideoProcessingMonitor, TemporaryVideoFile


class VideoProcessingMonitorSerializer(serializers.ModelSerializer):
    status_name = serializers.SerializerMethodField()
    uploaded_at = serializers.SerializerMethodField()
    upload_progress_percents_str = serializers.SerializerMethodField()

    class Meta:
        model = VideoProcessingMonitor
        fields = [
            'pk', 'video_name', 'status_name', 'has_slices',
            'uploaded_at', 'upload_progress_percents_str', 'events'
        ]

    def get_status_name(self, obj):
        return VideoProcessingMonitor.StatusChoices.get_choice(obj.status).label

    def get_uploaded_at(self, obj):
        return obj.created_at.strftime('%b. %d, %Y, %-I:%M %p')

    def get_upload_progress_percents_str(self, obj):
        upp = obj.upload_progress_percents
        ans = int(upp) if upp == int(upp) else upp
        return "{}%".format(ans)


class TemporaryVideoFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = TemporaryVideoFile
        fields = ['video', 'video_name']
