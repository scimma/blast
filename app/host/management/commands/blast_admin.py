"""
This module defines a custom Django management command "blast_admin". Its purpose is to
aid Blast admin and development operations that require the Django environment, such as
accessing database objects using the Django ORM.

The usage syntax is as follows:

   python manage.py blast_admin [func_name]

where "func_name()" is defined in either "util.py" (public code) or "local_util.py"
(local scratch ignored by Git).
"""
from django.core.management.base import BaseCommand
from host.log import get_logger
logger = get_logger(__name__)
from .util import *  # noqa: F401,F403
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
