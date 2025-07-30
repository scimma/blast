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

def convert_local_path_to_object_key(local_file_path:str, data_dir_name:str):
    """
    Convert paths of the format 'cutout_cdn/2010H/SDSS/SDSS_g.fits' to '2010H/v0.0.0/workflow_default/cutout_cdn/SDSS/SDSS_g.fits'
    """
    raw_path_components = local_file_path.replace(f'{data_dir_name}/', '').split("/")
    return f"{raw_path_components[0]}/v0.0.0/workflow_default/{data_dir_name}/{'/'.join(raw_path_components[1:])}"

def verify_data_integrity(download=False):
    '''Verify integrity of initial data file set'''
    s3_init = ObjectStore(conf=DATA_INIT_S3_CONF)
    s3_data = ObjectStore()
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
            s3_init.download_object(
                path=os.path.join('init/data', bucket_path),
                file_path=file_path,
                version_id=data_object['version_id'])
        etag = data_object['etag']
        size = data_object['size']
        # logger.debug(f'source etag: {etag}')
        checksum_match = s3_init.etag_compare(file_path, etag, size)
        log_msg = f'''Comparing "{file_path}"... {checksum_match}'''
        if checksum_match:
            logger.debug(log_msg)
        else:
            logger.error(log_msg)
            if not download:
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
                logger.info(f'''Downloaded file "{bucket_path}" passes integrity check.''')
        logger.debug(f'''Checking if "{bucket_path}" needs to be uploaded to bucket...''')
        for data_dir_name in [
            'cutout_cdn',
            'sed_output'
        ]:
            if bucket_path.startswith(f'{data_dir_name}/'):
                # Upload file to bucket and delete local copy
                object_key = os.path.join(
                    S3_BASE_PATH.strip('/'),
                    convert_local_path_to_object_key(bucket_path, data_dir_name))
                if not s3_data.object_exists(object_key):
                    logger.info(f'''Uploading file "{bucket_path}" to "{object_key}"...''')
                    s3_data.put_object(path=object_key, file_path=file_path)
                else:
                    logger.debug(f'''Object already in bucket: "{object_key}"''')
                assert s3_data.object_exists(object_key)


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
