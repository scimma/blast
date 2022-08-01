Task Runners
============

This page walks you through how to write a blast TaskRunner. A TaskRunner
performs a computational task in blast. There are two types of TaskRunner
in blast. The first is the TransientTaskRunner, which performs a computation on
a transient in the blast database (e.g., matching a host galaxy), and a
SystemTaskRunner which performs a system level task not related to a specific
transient (e.g, ingest a batch of transients from TNS or clean data directories).

Writing your own TaskRunner in blast should be straight forward if you inherit
from TransientTaskRunner or SystemTaskRunner. These classes handle of the
house keeping associated with running blast task, allowing you to just write the
task code.

This pages explains how to write your own TransientTaskRunner and
SystemTaskRunner.

Transient Task
--------------

A new :code:`TransientTaskRunner` should be implemented in
the :code:`app/host/transient_tasks.py` module. We will now explain how to
implement a :code:`TransientTaskRunner`.


Process method
^^^^^^^^^^^^^^

The _run_process method is where your task's code should go. This method
should contain all the necessary computations and saves to the database for your
task to be completed. It takes a Transient object as an argument and must return
a status message which indicates the status of the task after computation. As an
example, let's implement a simple task that just prints 'processing' and
then returns the processed status message.

.. code:: python

    def _run_process(transient):
        print('processing')
        return = "processed"

.. note::

    The available status messages can be found in
    :code:`app/host/fixtures/initial/setup_status.yaml`. The :code:`_run_process`
    method must return a string that matches the message field of
    one of the statuses in :code:`app/host/fixtures/initial/setup_status.yaml`.
    If you want to use a new status add it to
    :code:`app/host/fixtures/initial/setup_status.yaml`

Task name
^^^^^^^^^

The TaskRunner needs to specify which task it operates on. This is done through
implementing the task_name property. This methods takes no arguments and returns
a string which is the name of the task. Let's say we are implementing a
TransientTaskRunner that matches a transient to a host galaxy, this
TransientTaskRunner will alter the status of the Host match Task,

.. code:: python

    @property
    def task_name():
        return 'Host match'

.. note::

    You need to add your new task and its name into
    :code:`app/host/fixtures/initial/setup_tasks.yaml` making sure the return of
    task_name matches the name field in the fixture. This will ensure blast
    registers your task on start up.


Prerequisites
^^^^^^^^^^^^^

The TaskRunner needs to also specify which tasks need to be completed before it
should be run. This is done through implementing the _prerequisites method. This
function tasks no arguments and should return a dictionary with the name and
status of prerequisite tasks. For example, if before running your task you need
the Host match task to have status "not processed" and the Cutout download task
to have status "processed", it would look like this.

.. code:: python

    def _prerequisites():
        return {'Host match': 'not processed', 'Cutout download': 'processed'}

This ensures that your TaskRunner will only run on transients in the blast
database meeting the prerequisites.

.. note::

    The available tasks can be found in
    :code:`app/host/fixtures/initial/setup_tasks.yaml`.  The :code:`_prerequisites` method must
    return a dictionary with keys that match the name field of one of the tasks in
    :code:`app/host/fixtures/initial/setup_tasks.yaml` and values that match a
    status :code:`app/host/fixtures/initial/setup_status.yaml`.

Failed Status
^^^^^^^^^^^^^

You can specify what status happens if your :code:`_run_process` code
throws and exception and fails. This is done by implementing the
:code:`_failed_status_message method`.  This method takes no arguments and returns a
string which is the message of the failed status. Let's say we want the failed
status to be the Status with the message "failed",

.. code:: python

    def _failed_status_message()
        return "failed"

If you do not implement this method it will default to a "failed" status.

.. note::

    The available status messages can be found in
    :code:`app/host/fixtures/initial/setup_status.yaml`. The _failed_status_message
    method must return a string that matches the message field of one of the statuses in
    :code:`app/host/fixtures/initial/setup_status.yaml`. If you want to use a new
    status add it to :code:`app/host/fixtures/initial/setup_status.yaml`

Task Frequency
^^^^^^^^^^^^^^

You can specify the frequency at which as task should be run blast by implementing
the :code:`task_frequency_seconds` property. This function must return a positive
integer. If you do not implement this method, it will default to 60 seconds.

.. code:: python

    @property
    def task_frequency_seconds(self):
        return 60.0


Run on start up
^^^^^^^^^^^^^^^

You can specify whether your task runs periodically on start up of blast or needs
to be explicitly trigger from the djano admin by implementing
the :code:`task_initially_enabled` property. If you do not implement this method
it will default to true, meaning that the task will launch automatically on
startup.

.. code:: python

    @property
    def task_initially_enabled(self):
        """Will the task be run on start up"""
        return True


Full example class
^^^^^^^^^^^^^^^^^^

Putting this all together, the example :code:`TransientTaskRunner` class would be,

.. code:: python

    from .base_tasks import TransientTaskRunner

    class ExampleTaskRunner(TransientTaskRunner):
        """An Example TaskRunner"""

        def _run_process(transient):
            print('processing')
            return = "processed"

        def _prerequisites():
            return {"Host match": "not processed", "Cutout download": "processed"}

        @property
        def task_name():
            return "Host match"

        @property
        def task_frequency_seconds(self):
            return 60.0

        @property
        def task_initially_enabled(self):
            return True

        def _failed_status_message()
            return "failed"




System Task
-----------

The :code:`SystemTaskRunner` is somewhat simpler to implement as there is no chaining
of prerequisite tasks, and the results do not need to be displayed in the blast
web interface. New system task runners should be implemented in
the :code:`app/host/system_tasks.py` module. A full :code:`SystemTaskRunner`
would look like:

.. code:: python

    from .base_tasks import SystemTaskRunner

    class ExampleTaskRunner(SystemTaskRunner):
        """An Example TaskRunner"""

        @property
        def task_frequency_seconds(self):
            return 60.0

        @property
        def task_initially_enabled(self):
            return True

        def run_process(transient):
            #Put your code here!
            return = "processed"


Registering your task
---------------------

For blast to actually run your task you have to register it within the app. For
both a :code:`SystemTaskRunner` and a :code:`TransientTaskRunner` you have to
add an instance of your Taskrunner to the :code:`periodic_tasks`
list in :code:`app/host/task.py`.

To check that your task has been registered and is being run in blast go to
`<http://0.0.0.0:8000/admin/>`_ login and then go to `<http://0.0.0.0:8000/admin/django_celery_beat/periodictask/>`_
and you should see your task and its schedule.

You can check if your task is running without error by going to the flower
dashboard at `<http://0.0.0.0:8888>`_.
