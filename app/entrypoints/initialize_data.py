import os
import sys
from pathlib import Path
import json
# TODO: Redundant definitions of global config variables should be avoided, but we
#       cannot currently run in a Django shell *before* initializing the data files.
# from django.conf import settings
CUTOUT_ROOT = os.getenv("CUTOUT_ROOT", "/data/cutout_cdn")
SED_OUTPUT_ROOT = os.getenv("SED_OUTPUT_ROOT", "/data/sed_output")
S3_BASE_PATH = os.getenv("S3_BASE_PATH", "")

sys.path.append(os.path.join(str(Path(__file__).resolve().parent.parent)))
from host.object_store import ObjectStore
from host.log import get_logger
logger = get_logger(__name__)

DATA_INIT_S3_CONF = {
    'endpoint-url': os.getenv("S3_ENDPOINT_URL_INIT", 'https://js2.jetstream-cloud.org:8001'),
    'region-name': os.getenv("S3_REGION_NAME_INIT", ''),
    'aws_access_key_id': os.getenv("S3_ACCESS_KEY_ID_INIT", ''),
    'aws_secret_access_key': os.getenv("S3_SECRET_ACCESS_KEY_INIT", ''),
    'bucket': os.getenv("S3_BUCKET_INIT", 'blast-astro-data'),
}


def generate_file_manifest():
    '''Collect metadata for the latest versions of the objects in a JSON file'''
    s3init = ObjectStore(conf=DATA_INIT_S3_CONF)
    root_path = 'init/data/'
    objs = s3init.get_directory_objects(root_path)
    file_info = []
    for obj in objs:
        info = {
            'path': obj.object_name.replace(root_path, ''),
            'version_id': obj.version_id,
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
    s3_init = ObjectStore(conf=DATA_INIT_S3_CONF)
    s3_data = ObjectStore()
    data_root_dir = os.getenv('DATA_ROOT_DIR', '/mnt/data')
    with open(os.path.join(Path(__file__).resolve().parent, 'blast-data.json'), 'r') as fh:
        data_objects = json.load(fh)
    for idx, data_object in enumerate(data_objects):
        bucket_path = data_object['path']
        etag = data_object['etag']
        size = data_object['size']
        logger.debug(f'''[{idx + 1}/{len(data_objects)}] Processing "{bucket_path}"...''')
        object_processed = False
        for data_dir_name, data_root_path in [
            ('cutout_cdn', CUTOUT_ROOT),
            ('sed_output', SED_OUTPUT_ROOT)
        ]:
            # If the file should be stored in the dataset storage bucket, compare the etags directly
            if bucket_path.startswith(f'{data_dir_name}/'):
                object_key = os.path.join(
                    S3_BASE_PATH.strip('/'),
                    data_root_path.strip('/'),
                    bucket_path.replace(f'{data_dir_name}/', ''))
                existing_obj_etag = ''
                if s3_data.object_exists(object_key):
                    # If the object already exists, verify the checksum
                    existing_obj = s3_data.object_info(object_key)
                    existing_obj_etag = existing_obj.etag

                if existing_obj_etag == etag:
                    logger.debug(f'''Object already in bucket: "{object_key}"''')
                else:
                    # Upload file to bucket if it is missing and delete local copy
                    logger.info(f'''Object needs to be installed: "{object_key}"''')
                    logger.debug(f'''"{object_key}"       source etag: {etag}''')
                    logger.debug(f'''"{object_key}"    installed etag: {existing_obj_etag}''')
                    try:
                        if not download:
                            logger.warning('''Download disabled. Exiting.''')
                            sys.exit(1)
                        logger.info(f'''Downloading file "{bucket_path}"...''')
                        local_tmp_path = os.path.join('/tmp', bucket_path.strip('/').replace('/', '__'))
                        s3_init.download_object(
                            path=os.path.join('init/data', bucket_path),
                            file_path=local_tmp_path,
                            version_id=data_object['version_id'])
                        logger.info(f'''Uploading file "{bucket_path}" to "{object_key}"...''')
                        s3_data.put_object(path=object_key, file_path=local_tmp_path)
                        assert s3_data.object_exists(object_key)
                    finally:
                        # Delete FITS file from local file cache
                        os.remove(local_tmp_path)
                # Mark object as processed so it is not treated as a locally-installed file
                object_processed = True
        if object_processed:
            # Continue to next data_object
            continue
        # If the file should be stored in the shared mounted volume, download and/or compare with the official version
        file_path = os.path.join(data_root_dir, bucket_path)
        if not os.path.isfile(file_path):
            logger.info(f'''Missing file: {bucket_path}''')
            if not download:
                logger.warning('''Download disabled. Exiting.''')
                sys.exit(1)
            logger.debug(f'''Downloading file "{bucket_path}"...''')
            s3_init.download_object(
                path=os.path.join('init/data', bucket_path),
                file_path=file_path,
                version_id=data_object['version_id'])
        # logger.debug(f'source etag: {etag}')
        checksum_match = s3_init.etag_compare(file_path, etag, size)
        log_msg = f'''Comparing "{file_path}"... {checksum_match}'''
        if checksum_match:
            logger.debug(log_msg)
        else:
            logger.error(log_msg)
            if not download:
                logger.warning('''Download disabled. Exiting.''')
                sys.exit(1)
            logger.info(f'''Downloading file "{bucket_path}"...''')
            s3_init.download_object(
                path=os.path.join('init/data', bucket_path),
                file_path=file_path,
                version_id=data_object['version_id'])
            checksum_match = s3_init.etag_compare(file_path, etag, size)
            log_msg = f'''Comparing "{file_path}"... {checksum_match}'''
            if not checksum_match:
                logger.error(f'''Downloaded file "{bucket_path}" fails integrity check.''')
                sys.exit(1)
            else:
                logger.debug(f'''Downloaded file "{bucket_path}" passes integrity check.''')


if __name__ == '__main__':

    cmd = 'download'
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
    logger.debug(f'initialize_data.py command: {cmd}')
    if cmd == 'verify':
        # Verify uploads against local files
        verify_data_integrity(download=False)
    if cmd == 'download':
        # Verify uploads against local files
        verify_data_integrity(download=True)
    elif cmd == 'manifest':
        generate_file_manifest()
