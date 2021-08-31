from django.core.management.base import BaseCommand, CommandError
from videos.tasks import fetch_processing_videos_stats
from videos.models import VideoProcessingMonitor


class Command(BaseCommand):
    help = 'Refill video statuses db'

    def handle(self, *args, **options):
        # Make all VPMs Done
        self.stdout.write("Setting all existing video statuses as Done so they will be hidden in Processing Page")
        VideoProcessingMonitor.objects.all().update(
            status=VideoProcessingMonitor.StatusChoices.Done
        )
        # Fetch them from redis
        self.stdout.write("Updating video stats from redis")
        fetch_processing_videos_stats()

        self.stdout.write(self.style.SUCCESS('DONE'))
