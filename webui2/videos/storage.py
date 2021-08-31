import os
from storages.backends.s3boto3 import S3Boto3Storage

from django.core.files.storage import FileSystemStorage
from django.conf import settings


class OverwriteStorage(FileSystemStorage):

    def get_available_name(self, name, max_length=None):
        if self.exists(name):
            os.remove(os.path.join(str(settings.MEDIA_ROOT), name))
        return name


class PublicMediaStorage(S3Boto3Storage):
    location = 'media'
    default_acl = 'public-read'
    file_overwrite = True

    def __init__(self, progress_percentage_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_percentage_class = progress_percentage_class

    def _save(self, name, content):
        cleaned_name = self._clean_name(name)
        name = self._normalize_name(cleaned_name)
        params = self._get_write_parameters(name, content)

        if (self.gzip and
                params['ContentType'] in self.gzip_content_types and
                'ContentEncoding' not in params):
            content = self._compress_content(content)
            params['ContentEncoding'] = 'gzip'

        obj = self.bucket.Object(name)

        content.seek(0, os.SEEK_END)
        file_size = content.tell()
        filename = content.name

        content.seek(0, os.SEEK_SET)
        obj.upload_fileobj(
            content,
            ExtraArgs=params,
            Callback=self.progress_percentage_class(
                filename,
                file_size
            )
        )
        return cleaned_name
