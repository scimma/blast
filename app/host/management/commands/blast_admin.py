"""
This module defines a custom Django management command "blast_admin". Its purpose is to
aid Blast admin and development operations that require the Django environment, such as
accessing database objects using the Django ORM.

The usage syntax is as follows:

   python manage.py blast_admin [func_name] --input_args '{"arg1": "val1", "arg2": "val2"}'

where "func_name()" is defined in either "util.py" (public code) or "local_util.py"
(local scratch ignored by Git), and --input_args is a JSON-formatted string containing either
a list of scalar values to pass as positional arguments to "func_name" or a dictionary to be
passed as keyword arguments.
"""
from django.core.management.base import BaseCommand, CommandError
import json
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
        parser.add_argument('func_name', type=str,
                            help="Fully-qualified function name to call, e.g. 'myapp.utils.process_item'")
        parser.add_argument('--input_args', type=str, default='[]',
                            help=(
                                "JSON string representing the function arguments. "
                                "Use a list for positional args or an object for keyword args. "
                            ))

    def handle(self, *args, **options):
        func = options['func_name']

        # Parse the JSON argument payload.
        # Accept either a list (positional args) or dict (keyword args).
        raw_args = options['input_args']
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as err:
            raise CommandError(f"Could not parse JSON args: {err}")

        if isinstance(parsed, list):
            call_args = parsed
            call_kwargs = {}
        elif isinstance(parsed, dict):
            # Decide: if keys are numeric-string and user intended positional args, they should pass a list.
            call_args = []
            call_kwargs = parsed
        else:
            # Single scalar -> pass as single positional arg
            call_args = [parsed]
            call_kwargs = {}

        eval(f'''{func}(*{call_args}, **{call_kwargs})''')
