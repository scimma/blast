from django.core.management.base import BaseCommand
from host.log import get_logger
logger = get_logger(__name__)
try:
    from .local_util import *  # noqa: F401,F403
    scratch_module_exists = True
except ModuleNotFoundError:
    scratch_module_exists = False


class Command(BaseCommand):
    help = "Run scratch function"

    def add_arguments(self, parser):
        parser.add_argument("func_name", type=str)

    def handle(self, *args, **options):
        try:
            eval(f'''{options['func_name']}()''')
        except NameError as err:
            logger.error(err)
            if not scratch_module_exists:
                logger.error('''Custom functions must be defined in a local '''
                             '''"app/host/management/commands/local_util.py" module.''')
