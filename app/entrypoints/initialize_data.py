import os
import sys
from pathlib import Path
import json
import logging
import hashlib
from minio import Minio
import re

# Configure logging
logging.basicConfig(format='%(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', logging.INFO))


class ObjectStore:
    def __init__(self) -> None:
        '''Initialize S3 client'''
        self.config = {
            'endpoint-url': os.getenv("S3_ENDPOINT_URL", "https://js2.jetstream-cloud.org:8001"),
            'region-name': os.getenv("S3_REGION_NAME", "null"),
            'aws_access_key_id': os.getenv("ACCESS_KEY_ID"),
            'aws_secret_access_key': os.getenv("SECRET_ACCESS_KEY"),
            'bucket': os.getenv("S3_BUCKET", "blast-astro-data"),
        }
        self.bucket = self.config['bucket']
        self.client = None
        # If endpoint URL is empty, do not attempt to initialize a client
        if not self.config['endpoint-url']:
            return
        if self.config['endpoint-url'].find('https://'):
            secure = False
            endpoint = self.config['endpoint-url'].replace('http://', '')
        elif self.config['endpoint-url'].find('http://'):
            secure = True
            endpoint = self.config['endpoint-url'].replace('https://', '')

        self.client = Minio(
            endpoint=endpoint,
            access_key=self.config['aws_access_key_id'],
            secret_key=self.config['aws_secret_access_key'],
            region=self.config['region-name'],
            secure=secure,
        )

    def get_directory(self, root_path):
        objects = self.client.list_objects(
            bucket_name=self.bucket,
            prefix=root_path,
            include_version=True,
            recursive=True,
        )
        return [obj for obj in objects]

    def object_info(self, path):
        response = self.client.stat_object(
            bucket_name=self.bucket,
            object_name=path)
        return response

    def download_object(self, path="", file_path=""):
        self.client.fget_object(
            bucket_name=self.bucket,
            object_name=path,
            file_path=file_path,
        )

    def md5_checksum(self, file_path):
        '''https://stackoverflow.com/a/58239738'''
        m = hashlib.md5()
        with open(file_path, 'rb') as fh:
            for data in iter(lambda: fh.read(1024 * 1024), b''):
                m.update(data)
        hexdigest = m.hexdigest()
        logger.debug(f'calculated md5 checksum: {hexdigest}')
        return hexdigest

    def etag_checksum(self, file_path, etag_parts=1, file_size=0):
        '''https://stackoverflow.com/a/58239738'''
        md5s = []
        min_chunk_size = 16 * 1024**2
        chunk_size = int(file_size / etag_parts)
        if etag_parts == 1:
            chunk_size = file_size
        elif chunk_size < min_chunk_size:
            chunk_size = min_chunk_size
        chunk_size_mib = int(chunk_size / 1024**2)
        file_size_mib = int(file_size / 1024**2)
        logger.debug(f"chunk_size is {chunk_size_mib} MiB for file size {file_size_mib} bytes (etag parts: {etag_parts})")
        with open(file_path, 'rb') as fh:
            for data in iter(lambda: fh.read(chunk_size), b''):
                md5s.append(hashlib.md5(data).digest())
        digests_md5 = hashlib.md5(b''.join(md5s))
        etag_checksum = f'{digests_md5.hexdigest()}-{etag_parts}'
        logger.debug(f'calculated etag: {etag_checksum}')
        return etag_checksum

    def etag_compare(self, file_path, etag_source, file_size):
        '''https://stackoverflow.com/a/58239738'''
        etag_source = etag_source.strip('"')
        etag_local = ''
        if '-' in etag_source:
            etag_parts = int(re.search(r'^.+-([0-9]+$)', etag_source).group(1))
            etag_local = self.etag_checksum(file_path, etag_parts=etag_parts, file_size=file_size)
        elif '-' not in etag_source:
            etag_local = self.md5_checksum(file_path)
        if etag_source == etag_local:
            return True
        else:
            logger.warning(f'    source etag checksum: {etag_source}')
            logger.warning(f'     local etag checksum: {etag_local}')
        return False


def gather_file_data():
    '''Collect metadata for the latest versions of the objects in a JSON file'''
    s3 = ObjectStore()
    root_path = 'init/data/'
    objs = s3.get_directory(root_path)
    file_info = []
    for obj in objs:
        info = {
            'path': obj.object_name.replace(root_path, ''),
            'versionId': obj.version_id,
            'etag': obj.etag,
            'size': obj.size,
        }
        if obj.is_latest.lower() == "true":
            file_info.append(info)
        else:
            logger.debug(info)
    with open(os.path.join(Path(__file__).resolve().parent, 'blast-data.json'), 'w') as fh:
        json.dump(file_info, fh, indent=2)


def verify_data_integrity(download=False):
    '''Verify integrity of initial data file set'''
    s3 = ObjectStore()
    data_root_dir = os.getenv('DATA_ROOT_DIR', '/mnt/data')
    with open(os.path.join(Path(__file__).resolve().parent, 'blast-data.json'), 'r') as fh:
        data_objects = json.load(fh)
    for data_object in data_objects:
        bucket_path = data_object['path']
        file_path = os.path.join(data_root_dir, bucket_path)
        if not os.path.isfile(file_path):
            logger.error(f'''Missing file: {bucket_path}''')
            if not download:
                sys.exit(1)
            logger.info(f'''Downloading file "{bucket_path}"...''')
            s3.download_object(path=os.path.join('init/data', bucket_path), file_path=file_path)
        etag = data_object['etag']
        size = data_object['size']
        # logger.debug(f'source etag: {etag}')
        checksum_match = s3.etag_compare(file_path, etag, size)
        log_msg = f'''Comparing "{file_path}"... {checksum_match}'''
        if checksum_match:
            logger.debug(log_msg)
        else:
            logger.error(log_msg)
            if not download:
                sys.exit(1)
            logger.info(f'''Downloading file "{bucket_path}"...''')
            s3.download_object(path=os.path.join('init/data', bucket_path), file_path=file_path)
            checksum_match = s3.etag_compare(file_path, etag, size)
            log_msg = f'''Comparing "{file_path}"... {checksum_match}'''
            if not checksum_match:
                logger.error(f'''Downloaded file "{bucket_path}" fails integrity check.''')
                sys.exit(1)
            else:
                logger.info(f'''Downloaded file "{bucket_path}" passes integrity check.''')


if __name__ == '__main__':

    cmd = ''
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
    if cmd == 'verify':
        # Verify uploads against local files
        verify_data_integrity(download=False)
    if cmd == 'download':
        # Verify uploads against local files
        verify_data_integrity(download=True)
    elif cmd == 'gather':
        gather_file_data()
    sys.exit()
