# See discussion at https://github.com/celery/celery/issues/4079#issuecomment-1128954283
# for the origin of this code module.
from pathlib import Path

from celery import bootsteps
from celery.signals import worker_ready, worker_shutdown

HEARTBEAT_FILE = Path("/tmp/worker_heartbeat")
READINESS_FILE = Path("/tmp/worker_ready")


class LivenessProbe(bootsteps.StartStopStep):
    requires = {'celery.worker.components:Timer'}

    def __init__(self, worker, **kwargs):
        self.requests = []
        self.tref = None

    def start(self, worker):
        self.tref = worker.timer.call_repeatedly(
            1.0, self.update_heartbeat_file, (worker,), priority=10,
        )

    def stop(self, worker):
        HEARTBEAT_FILE.unlink(missing_ok=True)

    def update_heartbeat_file(self, worker):
        HEARTBEAT_FILE.touch()


@worker_ready.connect
def worker_ready(**_):
    READINESS_FILE.touch()


@worker_shutdown.connect
def worker_shutdown(**_):
    READINESS_FILE.unlink(missing_ok=True)
