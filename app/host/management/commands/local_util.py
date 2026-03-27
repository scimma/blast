
def celery_beat():
    from django_celery_beat.models import PeriodicTask, IntervalSchedule

    search = PeriodicTask.objects.all()
    # print(search)
    for task in search:
        # if task.name in [
        #     'Ingest missed TNS transients',
        #     'TNS data ingestion',
        # ]:
        #     task.delete()
        #     print(f'deleted: {task}')
        print(f'task: {task}')
        continue
        continue
        print(f'''{task} ({'enabled' if task.enabled else 'disabled'})''')
        if task.name == 'Delete GHOST files':
            task.delete()
        if task.name == 'Log transient processing status':
            task.delete()
            print(f'deleted: {task}')
        if task.name in [
            'Ingest missed TNS transients',
            'TNS data ingestion',
        ]:
            print(f'original: {task}')
            interval, created = IntervalSchedule.objects.get_or_create(period=IntervalSchedule.SECONDS,
                                                                       every=999999)
            task.interval = interval
            task.save()
            print(f'updated: {task}')

    return
    search = PeriodicTask.objects.filter(
        task='host.system_tasks.retrigger_failed_workflows')
    print(search)

    if search:
        search.delete()
        print('Periodic task deleted')
    else:
        print('Not found')


def check_thumbnail_status():
    # from host.transient_tasks import get_processing_status_and_progress
    import sys
    import os
    import json
    # from host.models import Transient, TaskRegister, Status
    # from host.transient_tasks import generate_thumbnail, generate_thumbnail_final
    # from host.transient_tasks import generate_thumbnail_sed_local, generate_thumbnail_sed_global
    from datetime import datetime, timezone

    def save_progress(force=False):
        if force or not idx % 100:
            print(f'''DEBUG: Dumping progress database to "{progress_db_obj_key}" ''')
            db['time_updated'] = str(datetime.now(timezone.utc))
            s3.put_object(data=db, path=progress_db_obj_key)
        elif not idx % 10:
            with open('/tmp/db.json', 'w') as temp_cache:
                db['time_updated'] = str(datetime.now(timezone.utc))
                json.dump(db, temp_cache)

    progress_db_basepath = 'apps/blast/tmp'
    progress_db_filename = 'thumbnail_check_db.json'
    progress_db_obj_key = os.path.join(progress_db_basepath, progress_db_filename)

    refresh_object_list = False
    refresh_lists = False
    local_file_path = os.path.join('/var/tmp', progress_db_filename)
    # local_file_path = ''

    if not local_file_path:
        from host.object_store import ObjectStore
        s3 = ObjectStore()

    db = {
        'time_updated': str(datetime.now(timezone.utc)),
        'obj_key_paths': [],
        'all_transients': [],
        'transients_with_cutout_thumbnails': [],
        'transients_without_cutout_thumbnails': [],
    }

    try:
        if local_file_path:
            with open(local_file_path) as fp:
                db = json.load(fp)
        elif s3.object_exists(path=progress_db_obj_key):
            progress_db_json = s3.get_object(path=progress_db_obj_key)
            db = json.loads(progress_db_json)
        if refresh_lists:
            db['transients_with_cutout_thumbnails'] = []
            db['transients_without_cutout_thumbnails'] = []
            from host.models import Transient
            all_transients = Transient.objects.all()
            num_transients = len(all_transients)
            print(f'''DEBUG: {num_transients} transients found.''')
            db['all_transients'] = [trans.name for trans in all_transients]
            db['time_updated'] = str(datetime.now(timezone.utc))
        if refresh_object_list:
            print('''DEBUG: Querying object list from S3 bucket...''')
            db['obj_key_paths'] = s3.list_directory('apps/blast/astro-data/data/cutout_cdn')
            db['time_updated'] = str(datetime.now(timezone.utc))
            print('''DEBUG: Query complete.''')
        if not local_file_path:
            print('''DEBUG: Storing progress to bucket...''')
            s3.put_object(data=db, path=progress_db_obj_key)
    except Exception as err:
        print(f'''ERROR: Error initializing progress DB: {err}''')
        sys.exit()

    print('''DEBUG: Querying transients objects...''')
    if refresh_lists:
        from host.models import Transient
        all_transients = Transient.objects.all()
        num_transients = len(all_transients)
        print(f'''DEBUG: {num_transients} transients found.''')
        db['all_transients'] = [trans.name for trans in all_transients]
        print('''DEBUG: Storing progress to bucket...''')
        db['time_updated'] = str(datetime.now(timezone.utc))
        if not local_file_path:
            s3.put_object(data=db, path=progress_db_obj_key)

    db['transients_with_cutout_thumbnails'] = []
    db['transients_without_cutout_thumbnails'] = []
    num_transients = len(db['all_transients'])
    last_trans_name = ''
    tr_idx = 0
    for idx, obj_key_path in enumerate(db['obj_key_paths']):
        curr_trans_name = obj_key_path.replace('apps/blast/astro-data/data/cutout_cdn/', '').split('/')[0]
        if curr_trans_name not in db['all_transients']:
            # If the transient is not in the Blast db, skip
            continue
        if curr_trans_name in db['transients_with_cutout_thumbnails'] + db['transients_without_cutout_thumbnails']:
            # If the current transient has already been categorized, continue to next path
            continue
        if curr_trans_name != last_trans_name:
            # A new transient is being examined
            if last_trans_name and last_trans_name not in db['transients_with_cutout_thumbnails']:
                print(f'''DEBUG: "{last_trans_name}" missing cutout JPG.''')
                if last_trans_name not in db['transients_without_cutout_thumbnails']:
                    db['transients_without_cutout_thumbnails'].append(last_trans_name)
            last_trans_name = curr_trans_name
            print(f'''DEBUG: ({tr_idx + 1}/{num_transients}) Analyzing "{curr_trans_name}"...''')
            tr_idx += 1
        if obj_key_path.lower().endswith('.jpg') and curr_trans_name not in db['transients_with_cutout_thumbnails']:
            db['transients_with_cutout_thumbnails'].append(curr_trans_name)

        if not local_file_path:
            save_progress()

    if not local_file_path:
        save_progress(force=True)

    with open(local_file_path, 'w') as fp:
        db['time_updated'] = str(datetime.now(timezone.utc))
        json.dump(db, fp)

    sys.exit()


def delete_transient_list():
    from host.host_utils import delete_transient
    import sys

    transients_to_delete = [
        '2025ass',
    ]

    # transients_to_delete = sys.argv[1:]

    print(f'''Deletion requested for {len(transients_to_delete)} transients.''')

    transients_deleted = []
    for transient_name in transients_to_delete:
        print(f'''Deleting transient "{transient_name}"...''')
        err_msg = delete_transient(transient_name)
        if err_msg:
            print(err_msg)
        else:
            transients_deleted.append(transient_name)
    for transient_name in [transient_name for transient_name in transients_to_delete
                           if transient_name not in transients_deleted]:
        print(f'''Unknown transient: "{transient_name}"...''')

    print(f'''Deleted {len(transients_deleted)} transients.''')


def generate_all_thumbnails():
    import yaml
    import sys
    from host.models import TaskRegister
    from host.models import Transient
    from host.models import Status
    from host.transient_tasks import generate_thumbnail
    from host.transient_tasks import generate_thumbnail_sed_local
    from host.transient_tasks import generate_thumbnail_sed_global
    from host.object_store import ObjectStore
    s3 = ObjectStore()

    # trans_obj_keys = s3.list_directory(f'/apps/blast/')
    # print(trans_obj_keys)
    # objects = s3.client.list_objects(
    #     bucket_name=s3.bucket,
    #     prefix='apps/',
    #     recursive=False,
    # )
    # print([obj.object_name for obj in objects])
    # sys.exit()

    year = '2025'
    batch_size = 22706
    dry_run = True
    dry_run = False
    dry_run_msg = '[dry-run] ' if dry_run else ''
    thumbnail_tasks = [
        'Generate thumbnail',
        # 'Generate thumbnail SED local',
        # 'Generate thumbnail SED global',
    ]
    transients = []
    print(f'''{dry_run_msg}DEBUG: Querying transients objects...''')
    transients = Transient.objects.filter(name__startswith=year)
    print(f'''{dry_run_msg}DEBUG: Querying task register objects...''')
    trs = TaskRegister.objects.all()
    print(f'''{dry_run_msg}DEBUG: Iterating over resultant objects...''')
    # for tr in trs.filter(transient__name__startswith=year):
    print(f'''{dry_run_msg}DEBUG: {len(transients)} transients found.''')
    for transient in transients[:batch_size]:
        # transient = tr.transient
        trans_name = transient.name
        print(f'''{dry_run_msg}DEBUG: Analyzing {trans_name}...''')
        for task_name in thumbnail_tasks:
            thumb_trs = trs.filter(transient__name=trans_name, task__name=task_name)
            if not thumb_trs:
                print(f'''{dry_run_msg}WARNING: "{trans_name}" missing task "{task_name}"''')
                continue
            for thumb_tr in thumb_trs:
                print(f'''{dry_run_msg}DEBUG: "{trans_name}" status "{task_name}" : {thumb_tr.status.message}''')
                run_task = False
                if thumb_tr.status.message != 'processed':
                    run_task = True
                    print(f'''{dry_run_msg}DEBUG: "{trans_name}" status is "{thumb_tr.status.message}".''')
                else:
                    # exists = s3.object_exists(thumbnail_object_key)
                    trans_obj_keys = s3.list_directory(f'apps/blast/astro-data/data/cutout_cdn/{trans_name}/')
                    # print(trans_obj_keys)
                    # If there are any JPEGs in the transient directory, assume one is the thumbnail
                    jpg_obj_keys = [key for key in trans_obj_keys if key.lower().endswith('.jpg')]
                    exists = True if jpg_obj_keys else False
                    print(f'''{dry_run_msg}DEBUG: "{trans_name}" JPEG files: {jpg_obj_keys}''')
                    run_task = not exists
                    if run_task:
                        print(f'''{dry_run_msg} DEBUG: "{trans_name}
                              " marked "processed" but JPG missing. Resetting task status...''')
                        if not dry_run:
                            thumb_tr.status = Status.objects.get(message='not processed')
                            thumb_tr.save()
                if run_task:
                    print(f'''{dry_run_msg}WARNING: "{trans_name}" needs to run "{task_name}".''')
                    if dry_run:
                        continue
                    if task_name == thumbnail_tasks[0]:
                        print(f'''{dry_run_msg}WARNING: "{trans_name}" launching task "{task_name}"...''')
                        generate_thumbnail.delay(trans_name)
                    # elif task_name == thumbnail_tasks[1]:
                    #     generate_thumbnail_sed_local.delay(trans_name)
                    # elif task_name == thumbnail_tasks[2]:
                    #     generate_thumbnail_sed_global.delay(trans_name)
                else:
                    print(f'''{dry_run_msg}DEBUG: "{trans_name}" thumbnail already generated.''')
        # if not trans_name.startswith(year):
        #     continue
        # if transient not in transients:
        #     transients.append(transient)
        #     print(f'''{dry_run_msg}DEBUG: Analyzing {trans_name}...''')
        #     for task_name in thumbnail_tasks:
        #         thumb_trs = trs.filter(transient__name=trans_name, task__name=task_name)
        #         if not thumb_trs:
        #             print(f'''{dry_run_msg}WARNING: {trans_name} missing task {task_name}''')
        #             continue
        #         if dry_run:
        #             continue
        #         for thumb_tr in thumb_trs:
        #             if thumb_tr.status.message != 'processed':
        #                 if task_name == thumbnail_tasks[0]:
        #                     generate_thumbnail.delay(trans_name)
        #                 elif task_name == thumbnail_tasks[1]:
        #                     generate_thumbnail_sed_local.delay(trans_name)
        #                 elif task_name == thumbnail_tasks[2]:
        #                     generate_thumbnail_sed_global.delay(trans_name)

    sys.exit()

    # for trans in Transient.objects.all()[:num_to_process]:
    for trans in Transient.objects.all():
        if not trans.name.startswith(year):
            continue
        print(f'''[{trans.name}] Analyzing tasks...''')
        tasks = []
        trs = TaskRegister.objects.filter(transient__name=trans.name)
        for tr in trs:
            # print(f'''[{tr.transient.name}] {tr.task.name}: {tr.status}''')
            tasks.append({
                'name': tr.task.name,
                'status': tr.status.message,
            })
        transients.append({
            'name': trans.name,
            'tasks': tasks,
        })

    print(yaml.dump(transients))
    print(f'Num transients: {len(transients)}')

    # sys.exit()

    with open(f'/var/tmp/all_transients.{year}.yaml', 'w') as fp:
        yaml.dump(transients, fp)

    num_tasks_to_run = 0

    for tr in transients:
        trans_name = tr['name']
        for task_name in thumbnail_tasks:
            if task_name not in [task['name'] for task in tr['tasks']]:
                print(f'''{trans_name} missing task {task_name}''')
                num_tasks_to_run += 1
                if dry_run:
                    continue
                if task_name == thumbnail_tasks[0]:
                    generate_thumbnail.delay(trans_name)
                # elif task_name == thumbnail_tasks[1]:
                #     generate_thumbnail_sed_local.delay(trans_name)
                # elif task_name == thumbnail_tasks[2]:
                #     generate_thumbnail_sed_global.delay(trans_name)
            else:
                taskreg = TaskRegister.objects.get(transient__name=trans_name, task__name=task_name)
                if taskreg.status.message != 'processed':
                    print(f'''{trans_name} needs to run {task_name} ({taskreg.status.message})''')
                    num_tasks_to_run += 1
                    if dry_run:
                        continue
                    if task_name == thumbnail_tasks[0]:
                        generate_thumbnail.delay(trans_name)
                    # elif task_name == thumbnail_tasks[1]:
                    #     generate_thumbnail_sed_local.delay(trans_name)
                    # elif task_name == thumbnail_tasks[2]:
                    #     generate_thumbnail_sed_global.delay(trans_name)

    print(f'Num tasks to process: {num_tasks_to_run}')


def queue():
    from app.celery import app

    print(app.amqp.queues)


def run_crop():
    import sys
    from datetime import datetime, timezone, timedelta
    from host.models import TaskRegister, Transient, Status
    from host.transient_tasks import crop_transient_images, generate_thumbnail_final
    from host.transient_tasks import get_processing_status_and_progress
    from host.tasks import retrigger_transient
    from django.db.models import F

    ########################################################################################################################
    transient_names = [
        '2025alln',
        '2021sgz',
        '2021afrz',
        '2021fnc',
    ]

    for transient_name in transient_names:
        tr = TaskRegister.objects.get(task__name='Crop transient images', transient__name=transient_name)
        tr.status = Status.objects.get(message='not processed')
        tr.save()
        print(f'''[{tr.transient.name}] {tr.task.name}: {tr.status}''')
        tr = TaskRegister.objects.get(task__name='Generate thumbnail final', transient__name=transient_name)
        tr.status = Status.objects.get(message='not processed')
        tr.save()
        print(f'''[{tr.transient.name}] {tr.task.name}: {tr.status}''')

        transient = Transient.objects.get(name__exact=transient_name)
        print(get_processing_status_and_progress(transient))
        transient.progress, transient.processing_status = get_processing_status_and_progress(transient)
        transient.save()
        # print(f'''Progress: {(transient.progress, transient.processing_status)}''')
        retrigger_transient(request=None, transient_name=transient_name)
    sys.exit()
    ########################################################################################################################

    ########################################################################################################################
    days_recent = 0
    dry_run = False
    target_transients = [
        "2025aijj",
    ]

    print('''DEBUG: Querying transients objects''')
    transients = Transient.objects.all().order_by(F('public_timestamp').desc(nulls_last=True))
    num_trans = len(transients)
    print(f'''DEBUG: {num_trans} transients found.''')
    print('''DEBUG: Querying task register objects''')
    trs = TaskRegister.objects.all()
    print(f'''DEBUG: {len(trs)} task register objects found.''')
    not_processed_status = Status.objects.get(message='not processed')
    processed_status = Status.objects.get(message='processed')
    curr_time = datetime.now(timezone.utc)
    for idx, transient in enumerate(transients):
        trans_name = transient.name
        public_timestamp = transient.public_timestamp
        if not public_timestamp:
            continue
        recent_transient = curr_time - public_timestamp <= timedelta(days=days_recent)
        if recent_transient:
            # print(f'''Recent transient: "{trans_name}": {public_timestamp}''')
            pass
        else:
            continue
        try:
            final_thumb_tr = trs.filter(transient__name=trans_name, task__name='Generate thumbnail final')[0]
            local_tr = trs.filter(transient__name=trans_name, task__name='Validate local photometry')[0]
            global_tr = trs.filter(transient__name=trans_name, task__name='Global aperture construction')[0]
            crop_tr = trs.filter(transient__name=trans_name, task__name='Crop transient images')[0]
            # if local_tr.status == processed_status and global_tr.status == processed_status:
            if global_tr.status == processed_status:
                if crop_tr.status == processed_status or trans_name in target_transients:
                    print(f'''[{idx}/{len(transients)}] "{trans_name}" ({public_timestamp}) Resetting '''
                          f''' "Crop transient images" and "Generate thumbnail final" tasks...''')
                    if not dry_run:
                        crop_tr.status = not_processed_status
                        crop_tr.save()
                        final_thumb_tr.status = not_processed_status
                        final_thumb_tr.save()
                        transient.progress, transient.processing_status = get_processing_status_and_progress(transient)
                        transient.save()
                        retrigger_transient(request=None, transient_name=trans_name)
                # else:
                #     print(f'''    ["{trans_name}"] Cropping not yet processed.''')
        except Exception as err:
            print(f'''    {err}''')
            pass
    sys.exit()
    ########################################################################################################################


def set_thumbnail_status_by_jpg_exists():
    from host.transient_tasks import get_processing_status_and_progress
    import sys
    import os
    import json
    from host.models import Transient, TaskRegister, Status
    from host.object_store import ObjectStore
    from host.transient_tasks import generate_thumbnail, generate_thumbnail_final
    from host.transient_tasks import generate_thumbnail_sed_local, generate_thumbnail_sed_global
    s3 = ObjectStore()

    def recalc_progress(transient):
        # The processing status should be calculated
        transient.progress, transient.processing_status = get_processing_status_and_progress(transient)
        transient.save()

    # class progress_db():
    #     def __init__(self):
    #         self.obj_key_paths = []
    #         self.transients_processed = transients_processed

    db = {
        'obj_key_paths': [],
        'transients_processed': [],
    }

    # start_idx = 1
    # refresh_object_list = False
    # refresh_object_list = True

    obj_key_file = '/var/tmp/all_trans_obj_keys.json'
    thumbnail_tasks = [
        ('Generate thumbnail', generate_thumbnail),
        ('Generate thumbnail final', generate_thumbnail_final),
        ('Generate thumbnail SED local', generate_thumbnail_sed_local),
        ('Generate thumbnail SED global', generate_thumbnail_sed_global),
    ]

    progress_db_basepath = 'apps/blast/tmp'
    progress_db_filename = 'thumbnail_task_update_db.json'
    progress_db_obj_key = os.path.join(progress_db_basepath, progress_db_filename)
    try:
        if s3.object_exists(path=progress_db_obj_key):
            progress_db_json = s3.get_object(path=progress_db_obj_key)
            db = json.loads(progress_db_json)
            if db['obj_key_paths']:
                db['obj_key_paths'] = []
        # # Always refresh file listing from object store. Slow but necessary.
        # if refresh_object_list or not db['obj_key_paths']:
        #     print('''DEBUG: Querying object list from S3 bucket...''')
        #     db['obj_key_paths'] = s3.list_directory('apps/blast/astro-data/data/')
        #     s3.put_object(data=db, path=progress_db_obj_key)
    except Exception as err:
        print(f'''ERROR: Error initializing progress DB: {err}''')
        sys.exit()

    print('''DEBUG: Querying transients objects...''')
    transients = Transient.objects.all()
    num_trans = len(transients)
    print(f'''DEBUG: {num_trans} transients found.''')
    remaining_transients = [transient for transient in transients if transient.name not in db['transients_processed']]
    num_remaining_transients = len(remaining_transients)
    print(f'''DEBUG: {num_remaining_transients} transients remaining to process.''')
    print('''DEBUG: Querying task register objects...''')
    trs = TaskRegister.objects.all()

    for idx, transient in enumerate(remaining_transients):
        trans_name = transient.name
        if trans_name in db['transients_processed']:
            continue
        print(f'''DEBUG: ({idx + 1}/{num_remaining_transients}) Analyzing "{trans_name}"...''')
        print('''DEBUG: Querying object list from S3 bucket...''')
        try:
            cutout_keys = s3.list_directory(f'apps/blast/astro-data/data/cutout_cdn/{trans_name}/')
        except Exception as err:
            print(f'''ERROR: Error listing cutout_cdn: {err}''')
            cutout_keys = []
        try:
            sed_keys = s3.list_directory(f'apps/blast/astro-data/data/sed_output/{trans_name}/')
        except Exception as err:
            print(f'''ERROR: Error listing sed_output: {err}''')
            sed_keys = []
        for task_name, task_func in thumbnail_tasks:
            thumb_trs = trs.filter(transient__name=trans_name, task__name=task_name)
            if not thumb_trs:
                print(f'''WARNING: "{trans_name}" missing task "{task_name}"''')
                recalc_progress(transient)
                continue
            for thumb_tr in thumb_trs:
                if thumb_tr.status.message != 'processed':
                    print(f'''DEBUG: "{trans_name}" status "{task_name}" : {thumb_tr.status.message}''')
                    continue
                if task_name == 'Generate thumbnail':
                    jpg_obj_keys = [key for key in cutout_keys
                                    if key.lower().endswith('.jpg')]
                elif task_name == 'Generate thumbnail SED local':
                    jpg_obj_keys = [key for key in sed_keys if key.lower().endswith('local.jpg')]
                elif task_name == 'Generate thumbnail SED global':
                    jpg_obj_keys = [key for key in sed_keys if key.lower().endswith('global.jpg')]
                print(f'''DEBUG: ["{trans_name}"] "{task_name}" JPEG files: {jpg_obj_keys}''')
                exists = True if jpg_obj_keys else False
                if not exists:
                    print(f'''DEBUG: ["{trans_name} "] task "{task_name}
                          " marked "processed" but JPG missing. Resetting status...''')
                    thumb_tr.status = Status.objects.get(message='not processed')
                    thumb_tr.save()
                    recalc_progress(transient)
                    task_func.delay(trans_name)
        if trans_name not in db['transients_processed']:
            db['transients_processed'].append(trans_name)
        if not idx % 100:
            print(f'''DEBUG: Dumping progress database to "{progress_db_obj_key}" ''')
            s3.put_object(data=db, path=progress_db_obj_key)
        elif not idx % 10:
            with open('/tmp/db.json', 'w') as temp_cache:
                json.dump(db, temp_cache)
    with open('/tmp/thumbnails_complete', 'w') as fp:
        fp.write('Complete')
    sys.exit()


def status_query():
    from app.celery import app
    import yaml
    import sys

    report = {
        'tasks': {}
    }

    inspect = app.control.inspect()
    all_tasks = []
    for inspect_func, task_type in [
            (inspect.active, 'active'),
            (inspect.scheduled, 'scheduled'),
            (inspect.reserved, 'reserved')]:
        try:
            items = inspect_func(safe=True).items()
            tasks = [task for worker, worker_tasks in items for task in worker_tasks]
            all_tasks.extend([{'name': task['name'],
                               'type': task_type,
                               'args': task['args'],
                               } for task in tasks])
        except Exception:
            continue
    report['tasks'] = all_tasks
    print(f'''{yaml.dump(report)}''')
    sys.exit(0)


def duplicate_taskregister_query():
    from host.models import Transient, TaskRegister
    import yaml

    report = {}
    report['tasks'] = {}

    def make_report(transient, workflow_tasks, registered_tasks):
        return {
            'registered_tasks': [{
                'name': task.task.name,
                'message': task.status.message,
            } for task in registered_tasks],
            'workflow_tasks': [{
                'name': task.task.name,
                'message': task.status.message,
            } for task in workflow_tasks],
        }

    trans_dups = {}
    trans_tasks = {}
    transients_with_dups = []
    for transient in Transient.objects.all():

        duplicate_tasks = []
        workflow_tasks = []
        registered_tasks = [task for task in TaskRegister.objects.filter(transient__name=transient.name)]
        # Warn if there are duplicates
        registered_task_names = [task.task.name for task in registered_tasks]
        if len(registered_task_names) > len(set(registered_task_names)):
            print(f'''"{transient.name}": duplicate registered tasks detected!''')
            transients_with_dups.append(transient.name)
            continue
        else:
            print(f'''"{transient.name}": clean.''')
            report['tasks'][transient.name] = make_report(
                transient, workflow_tasks=registered_tasks, registered_tasks=[])
            continue

        for registered_task in registered_tasks:
            # print(f'''"{transient.name}" registered task: {(registered_task.task.name, registered_task.status.type)}''')
            # If the registered task is a duplicate, replace the existing workflow_task item if the status is better
            existing_tasks = [task for task in workflow_tasks if task.task.name == registered_task.task.name]
            if not existing_tasks:
                # print(f'''"{transient.name}"     appending task: {(registered_task.task.name, registered_task.status.type)}''')
                workflow_tasks.append(registered_task)
                continue
            existing_task = existing_tasks[0]
            if existing_task.status.type == 'success':
                continue
            if registered_task.status.type == 'success':
                print(f'''"{transient.name}": better task found: {(registered_task.task.name, registered_task.status.type)}''')
                workflow_tasks.remove(existing_task)
                workflow_tasks.append(registered_task)
                continue
            # Prefer the error status because that means the task finished
            if existing_task.status.type != 'error' and registered_task.status.type == 'error':
                print(f'''"{transient.name}": better task found: {(registered_task.task.name, registered_task.status.type)}''')
                workflow_tasks.remove(existing_task)
                workflow_tasks.append(registered_task)
                continue
            # Otherwise, the existing task type is either "blank" or "warning" ("not processed" or "processing"),
            # in which case we would choose "processing"
            if existing_task.status.type == 'blank' and registered_task.status.type == 'warning':
                print(f'''"{transient.name}": better task found: {(registered_task.task.name, registered_task.status.type)}''')
                workflow_tasks.remove(existing_task)
                workflow_tasks.append(registered_task)
                continue
        report['tasks'][transient.name] = make_report(transient, workflow_tasks, registered_tasks)

    report['transients_with_dups'] = transients_with_dups

    # with open('/tmp/transient_dups.yaml', 'w') as fp:
    #     yaml.dump(report, fp)

    print(yaml.dump(report, sort_keys=False))


def transient_prune_duplicate_taskregister_objects():
    from host.models import Transient
    from host.models import TaskRegister
    import yaml
    import os

    def create_report(transient, tasks):
        report = {
            'name': transient.name,
            'tasks': [{
                'name': task.task.name,
                'last_modified': task.last_modified,
                'id': task.id,
                'transient_id': task.transient_id,
                'status': task.status.message,
            } for task in tasks],
        }
        # print(yaml.dump(report))
        return report

    delete_dups = os.getenv('DELETE_DUPLICATES', 'false').lower() == 'true'
    target_transients = [transient for transient in os.getenv('TARGET_TRANSIENTS', '').split(',') if transient]

    if isinstance(target_transients, list) and len(target_transients) > 0:
        print(f'''Target transients: {target_transients}''')
        transients_processing = Transient.objects.filter(processing_status="processing", name__in=target_transients)
    else:
        transients_processing = Transient.objects.filter(processing_status="processing")

    for transient in transients_processing:
        print(f'''Analyzing transient "{transient.name}"...''', end='')
        # print('''Identifying and deleting duplicates...''', end='')

        tasks = TaskRegister.objects.filter(transient__name=transient.name)
        report_before = create_report(transient, tasks)

        # Identify duplicate TaskRegister objects
        task_names = [task['name'] for task in report_before['tasks']]
        duplicate_task_names = [task_name for task_name in set(task_names) if task_names.count(task_name) > 1]
        for duplicate_task_name in duplicate_task_names:
            duplicate_tasks = [task for task in report_before['tasks'] if task['name'] == duplicate_task_name]
            print(f'''Duplicate TaskRegister objects ({len(duplicate_tasks)}): {duplicate_tasks}''')
            incomplete_tasks = [task for task in duplicate_tasks if task['status'] != "processed"]
            print(
                f'''    Incomplete TaskRegister objects in duplicate set ({len(incomplete_tasks)}): {incomplete_tasks}''')
            # If there are is at least one duplicate tasks with status "processed", delete the others
            if len(incomplete_tasks) < len(duplicate_tasks):
                for incomplete_task in incomplete_tasks:
                    incomplete_task_id = incomplete_task['id']
                    if delete_dups:
                        print(f'''    Deleting duplicate TaskRegister object with id = {incomplete_task_id}...''')
                        task_reg = TaskRegister.objects.get(id=incomplete_task_id)
                        task_reg.delete()
                    else:
                        print(
                            f'''    [dry-run] Deleting duplicate TaskRegister object with id = {incomplete_task_id}...''')

        if duplicate_task_names:
            print(' done. Duplicate registered tasks detected.')
            print('''Before deduplication''')
            report_before = create_report(transient, tasks)
            print(yaml.dump(report_before))
            print('''After deduplication''')
            tasks = TaskRegister.objects.filter(transient__name=transient.name)
            report_after = create_report(transient, tasks)
            print(yaml.dump(report_after))
        else:
            print(' done. No duplicate registered tasks.')

        # from host.base_tasks import get_processing_status
        # print('''Updating processing status...''')
        # transient.processing_status = get_processing_status(transient)
        # transient.save()
        # print(f'''"{transient.name}": {transient.processing_status}''')


def transient_stats():
    def download():
        from host.models import Transient
        import json
        import sys
        from datetime import datetime, timezone
        transients = Transient.objects.all()
        all_trans_data = []
        for transient in transients:
            try:
                trans_data = {
                    'name': transient.name,
                    'public_timestamp': transient.public_timestamp.isoformat() if isinstance(transient.public_timestamp, datetime) else None,
                    'processing_status': transient.processing_status,
                    'software_version': transient.software_version,
                    'tns_id': transient.tns_id,
                    'tns_prefix': transient.tns_prefix,
                    'host': {'name': transient.host.name} if transient.host else None,
                    'redshift': transient.redshift,
                    'spectroscopic_class': transient.spectroscopic_class,
                    'photometric_class': transient.photometric_class,
                    # 'milkyway_dust_reddening': transient.milkyway_dust_reddening,
                }
            except Exception as err:
                print(f'''Error parsing transient: {err}''')
            # print(json.dumps(trans_data, indent=2))
            all_trans_data.append(trans_data)

        report = {
            'date_collected': datetime.now(tz=timezone.utc).isoformat(),
            'transients': all_trans_data,
        }

        with open('/tmp/transients.json', 'w') as trans_file:
            json.dump(report, trans_file, indent=2)

    def order_by_time():
        import json
        import sys
        from datetime import datetime, timezone
        with open('/var/tmp/transients.json', 'r') as trans_file:
            report = json.load(trans_file)
        # report = {}
        # report['number_of_transients'] = len(report)
        transients = []
        for transient in report['transients']:
            if not transient['public_timestamp']:
                transient['public_timestamp'] = datetime.fromtimestamp(0, tz=timezone.utc)
            else:
                transient['public_timestamp'] = datetime.fromisoformat(transient['public_timestamp'])
            # print(transient['public_timestamp'])
            assert isinstance(transient['public_timestamp'], datetime)
            transient['public_timestamp'] = datetime.isoformat(transient['public_timestamp'])
            transients.append(transient)
        # Sort by public timestamp
        transients = sorted(transients, key=lambda tr: tr['public_timestamp'])
        report['transients'] = transients
        with open('/var/tmp/transients.sorted.json', 'w') as trans_file:
            json.dump(report, trans_file, indent=2)

    def bin_by_week():
        import json
        import sys
        from datetime import datetime, timedelta
        # Load time-ordered file
        with open('/var/tmp/transients.sorted.json', 'r') as trans_file:
            report = json.load(trans_file)
        trans_binned_by_week = {}
        start_time = None
        weekly_transients = []
        for transient in report['transients']:
            transient['public_timestamp'] = datetime.fromisoformat(transient['public_timestamp'])
            assert isinstance(transient['public_timestamp'], datetime)
            if transient['public_timestamp'] < datetime.fromisoformat('2024-01-01T00:00:00.000000+00:00'):
                continue
            if not start_time:
                start_time = transient['public_timestamp']
                transient['public_timestamp'] = datetime.isoformat(transient['public_timestamp'])
                weekly_transients = [transient]
                continue
            if start_time + timedelta(days=7) >= transient['public_timestamp']:
                # Convert back to time string
                transient['public_timestamp'] = datetime.isoformat(transient['public_timestamp'])
                weekly_transients.append(transient)
            else:
                trans_binned_by_week[start_time.isoformat()] = weekly_transients
                start_time = None
        # transients = []
        # for transient in report['transients']:
        #     transients.append(transient)

        report = trans_binned_by_week
        with open('/var/tmp/transients.weekly.json', 'w') as trans_file:
            json.dump(report, trans_file, indent=2)

    def analyze():
        import json
        import yaml
        import sys
        from datetime import datetime, timedelta
        # Load time-ordered file
        with open('/var/tmp/transients.weekly.json', 'r') as trans_file:
            trans_binned_by_week = json.load(trans_file)
        report = {}
        report['num_weeks'] = len(trans_binned_by_week)
        rows = []
        for week, transients in trans_binned_by_week.items():
            rows.append(f'''{datetime.fromisoformat(week).strftime('%Y/%m/%d')},{len(transients)}\n''')
        from statistics import mean
        report['avg_transients_per_week'] = round(
            mean([len(transients) for week, transients in trans_binned_by_week.items()]))
        print(yaml.dump(report))

        with open('/var/tmp/transients.weekly.csv', 'w') as trans_file:
            trans_file.writelines(rows)

    def plot():
        import csv
        import matplotlib.pyplot as plt
        import numpy as np
        weeks = []
        xvals = []
        num_transients = []
        idx = 1
        with open('/var/tmp/transients.weekly.csv', 'r') as trans_file:
            reader = csv.reader(trans_file)
            for row in reader:
                xvals.append(int(idx))
                weeks.append(row[0])
                num_transients.append(int(row[1]))
                idx += 1
        # plt.style.use('_mpl-gallery')
        fig, ax = plt.subplots(1, 1)
        ax.plot(xvals[:], num_transients[:], 'd', label='num_transients', linestyle='-', linewidth=2)
        ax.set_xticks(xvals[:], labels=weeks[:], rotation=70, ha="right", rotation_mode="anchor")
        ax.set_xlabel('Week')
        ax.set_ylabel('Number of transients')
        # plt.figure(figsize=(6.4, 4.8), dpi=50)
        # print(plt.figure().get_figheight())
        # print(plt.figure().get_figwidth())
        # plt.savefig('/var/tmp/transients_by_week.png')
        plt.rc('font', size=10)
        plt.show()

    # download()
    order_by_time()
    bin_by_week()
    analyze()
    plot()


def trigger_all_sed_thumbs():
    import sys
    from host.models import Transient, TaskRegister, Status
    from host.transient_tasks import generate_thumbnail_sed_local
    from host.transient_tasks import generate_thumbnail_sed_global

    thumbnail_tasks = [
        ('Generate thumbnail SED local', generate_thumbnail_sed_local),
        ('Generate thumbnail SED global', generate_thumbnail_sed_global),
    ]
    print('''DEBUG: Querying transients objects''')
    transients = Transient.objects.all()
    num_trans = len(transients)
    print(f'''DEBUG: {num_trans} transients found.''')
    print('''DEBUG: Querying task register objects''')
    trs = TaskRegister.objects.all()
    print(f'''DEBUG: {len(trs)} task register objects found.''')
    for idx, transient in enumerate(transients):
        trans_name = transient.name
        print(f'''DEBUG: ({idx + 1}/{num_trans}) Analyzing "{trans_name}"''')
        for task_name, task_func in thumbnail_tasks:
            thumb_trs = trs.filter(transient__name=trans_name, task__name=task_name)
            if not thumb_trs:
                print(f'''WARNING: "{trans_name}" missing task "{task_name}"''')
                continue
            for thumb_tr in thumb_trs:
                thumb_tr.status = Status.objects.get(message='not processed')
                thumb_tr.save()
                print(f'''DEBUG: "{trans_name}" launching task "{task_name}"''')
                task_func.delay(trans_name)
    sys.exit()


def usage_metrics():
    from host.models import UsageMetricsLog
    from host.system_tasks import usage_log_roller
    import sys

    for log in UsageMetricsLog.objects.all():
        print(
            f'''{log.request_time}, {log.request_method}, {log.request_url}, {log.request_user}, {log.request_ip},
            {log.submitted_data} ''')
    sys.exit()
    usage_log_roller.delay()


def api_test():

    from urllib.request import urlopen
    import json
    import yaml
    import sys

    def download_data(filepath):
        transient_name = sys.argv[1]
        # Download basic transient data
        url = f'https://blast.scimma.org/api/transient/?name={transient_name}&format=json'
        # Download all transient data
        url = f'https://blast.scimma.org/api/transient/get/{transient_name}?format=json'
        # Download all transients
        url = 'https://blast.scimma.org/api/transient'
        # print(url)
        response = urlopen(url)
        data = json.loads(response.read())
        with open(filepath, 'w') as fp:
            yaml.dump(data, fp)

    def analyze_data(filepath):
        with open(filepath, 'r') as fp:
            transients = yaml.safe_load(fp)
        print(f'''Number of transients (total): {len(transients)}''')
        print(f'''Number of transients (100%): {len([tr for tr in transients if tr['progress'] == 100])}''')
        print(f'''Number of transients (0%): {len([tr for tr in transients if tr['progress'] == 0])}''')
        print(f'''Number of transients (1-99%): {len([tr for tr in transients if tr['progress'] not in [0, 100]])}''')

    if __name__ == "__main__":
        download_data('temp.yaml')
        # analyze_data('/home/andrew/tmp/blast_data.202412161050.yaml')
