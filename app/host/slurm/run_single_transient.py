#!/usr/bin/env python
# D. Jones - 4/5/23
# django cron to be called by slurm
# should run a single transient all
# the way through.  data saved to sqlite
# and we can figure out how to export it
# later
import datetime
import os

import host.transient_tasks
from django_cron import CronJobBase
from django_cron import Schedule
from host import tasks
from host.models import Status
from host.models import Task
from host.models import TaskRegister
from host.models import Transient

transient_name = os.environ["BLAST_TRANSIENT_NAME"]
transient_ra = os.environ["BLAST_TRANSIENT_RA"]
transient_dec = os.environ["BLAST_TRANSIENT_DEC"]
try:
    transient_redshift = os.environ["BLAST_TRANSIENT_REDSHIFT"]
except Exception as e:
    transient_redshift = None

tasks_in_order = [
    "Transient MWEBV",
    "Host match",
    "Host MWEBV",
    "Cutout download",
    "Host information",
    "Transient information",
    "Global aperture construction",
    "Global aperture photometry",
    "Validate global photometry",
    "Local aperture photometry",
    "Validate local photometry",
    "Global host SED inference",
    "Local host SED inference",
]
tasks_classes_in_order = [
    host.transient_tasks.MWEBV_Transient(),
    host.transient_tasks.HostMatch(),
    host.transient_tasks.MWEBV_Host(),
    host.transient_tasks.ImageDownload(),
    host.transient_tasks.HostInformation(),
    host.transient_tasks.TransientInformation(),
    host.transient_tasks.GlobalApertureConstruction(),
    host.transient_tasks.GlobalAperturePhotometry(),
    host.transient_tasks.ValidateGlobalPhotometry(),
    host.transient_tasks.LocalAperturePhotometry(),
    host.transient_tasks.ValidateLocalPhotometry(),
    host.transient_tasks.GlobalHostSEDFitting(),
    host.transient_tasks.LocalHostSEDFitting(),
]

tasks_sed = [
    "Global host SED inference",
]
tasks_sed_classes = [
    host.transient_tasks.GlobalHostSEDFitting(),
]


class run_single(CronJobBase):
    RUN_EVERY_MINS = 3

    schedule = Schedule(run_every_mins=RUN_EVERY_MINS)
    code = "host.slurm.run_single_transient.run_single"

    def do(self):
        self.run()

    def run(self):
        transients = Transient.objects.filter(name=transient_name)
        if not len(transients):
            transient = Transient.objects.create(
                name=transient_name,
                ra_deg=transient_ra,
                dec_deg=transient_dec,
                tns_id=1,
                redshift=transient_redshift,
            )
        else:
            transient = transients[0]

        for to, tc in zip(tasks_in_order, tasks_classes_in_order):
            for t in tasks.periodic_tasks:
                if t.task_name == to:
                    task = Task.objects.get(name__exact=t.task_name)
                    task_register = TaskRegister.objects.all()
                    task_register = task_register.filter(transient=transient, task=task)

                    if not len(task_register):
                        task_register = TaskRegister.objects.create(
                            transient=transient,
                            task=task,
                            status=Status.objects.get(message="not processed"),
                            last_modified=datetime.datetime.now(),
                            last_processing_time_seconds=0,
                        )
                    else:
                        task_register = task_register[0]

                    # if task_register.status.message != 'processed':
                    task_register.status = Status.objects.get(message="not processed")
                    try:
                        status = t.run_process(task_register)
                    except Exception as e:
                        print(e)
                        # import pdb; pdb.set_trace()
                        # status = tc._run_process(transient)
                        # task_register.status = Status.objects.get(message=status)
                        # task_register.save()
                        raise e

                    break


class run_single_sed(CronJobBase):
    RUN_EVERY_MINS = 3

    schedule = Schedule(run_every_mins=RUN_EVERY_MINS)
    code = "host.slurm.run_single_transient.run_single"

    def do(self):
        self.run()

    def run(self):
        transients = Transient.objects.filter(name=transient_name)
        if not len(transients):
            transient = Transient.objects.create(
                name=transient_name,
                ra_deg=transient_ra,
                dec_deg=transient_dec,
                tns_id=1,
                redshift=transient_redshift,
            )
        else:
            transient = transients[0]

        for to, tc in zip(tasks_sed, tasks_sed_classes):
            for t in tasks.periodic_tasks:
                if t.task_name == to:
                    task = Task.objects.get(name__exact=t.task_name)
                    task_register = TaskRegister.objects.all()
                    task_register = task_register.filter(transient=transient, task=task)

                    if not len(task_register):
                        task_register = TaskRegister.objects.create(
                            transient=transient,
                            task=task,
                            status=Status.objects.get(message="not processed"),
                            last_modified=datetime.datetime.now(),
                            last_processing_time_seconds=0,
                        )
                    else:
                        task_register = task_register[0]

                    # if task_register.status.message != 'processed':
                    task_register.status = Status.objects.get(message="not processed")
                    try:
                        status = t.run_process(task_register, save=False)
                    except Exception as e:
                        print(e)
                        # import pdb; pdb.set_trace()
                        # status = tc._run_process(transient)
                        # task_register.status = Status.objects.get(message=status)
                        # task_register.save()
                        raise e

                    break
