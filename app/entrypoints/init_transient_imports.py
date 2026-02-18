import os
import sys
from pathlib import Path
from host.models import Transient

sys.path.append(os.path.join(str(Path(__file__).resolve().parent.parent)))
# from host.object_store import ObjectStore
from host.host_utils import import_transient_info
from host.log import get_logger
logger = get_logger(__name__)


def import_datasets(transient_name_list):
    transient_search = Transient.objects.filter(name__in=transient_name_list)
    for transient_name in transient_name_list:
        if transient_search.filter(name__exact=transient_name).exists():
            logger.debug(f'Skipping existing transient "{transient_name}"')
            continue
        logger.info(f'Installing transient dataset "{transient_name}"...')
        with open(f'''/data/transient_datasets/{transient_name}.tar.gz''', 'rb') as dataset_fileobj:
            # Ignore errors
            imported_transient_names, import_failures = import_transient_info(dataset_fileobj)
            for import_failure in import_failures:
                logger.error(f'''Failed to import "{import_failure['transient_name']}": '''
                             f'''"{import_failure['err_msg']}"''')


import_datasets([
    '2026dix',
    '2026dkf',
    '2026dgt',
])
