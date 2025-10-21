from storages.backends.s3boto3 import S3Boto3Storage
from django.conf import settings


class S3MediaStorage(S3Boto3Storage):
    location = "media"
    default_acl = "public-read"
    def __init__(self, *args, **kwargs):
        if settings.S3_ENDPOINT_URL:
            self.secure_urls = False
            self.custom_domain = settings.S3_ENDPOINT_URL
        super(S3MediaStorage, self).__init__(*args, **kwargs)