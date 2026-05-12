"""
This module contains functions designed to be invoked as arguments to the `dev`
custom Django management command. See the `dev.py` module docstring for usage instructions.
"""


def render_homepage():
    from host.views import update_home_page_statistics
    # Render the initial version of the static home page
    update_home_page_statistics()


def show_usage_log_buffer():
    from host.models import UsageMetricsLog
    logs = UsageMetricsLog.objects.all().order_by('request_time')
    print(logs)


def update_periodic_task():
    from django_celery_beat.models import PeriodicTask, IntervalSchedule

    search = PeriodicTask.objects.all()
    for task in search:
        print(f'task: {task}')
        if task.name in [
            # 'Usage log roller',
        ]:
            interval, created = IntervalSchedule.objects.get_or_create(period=IntervalSchedule.SECONDS,
                                                                       every=999999)
            task.interval = interval
            # task.save()
            print(f'updated: {task}')


def delete_transient_list(transients_to_delete=[]):
    """
    python manage.py dev delete_transient_list --input_args \
        '{"transients_to_delete": ["tran_1", "tran_2", ...]}'
    """
    from host.host_utils import delete_transient

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


def celery_task_queue_query():
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


def transient_stats():
    def download():
        from host.models import Transient
        import json
        from datetime import datetime, timezone
        transients = Transient.objects.all()
        all_trans_data = []
        for transient in transients:
            try:
                public_timestamp = None
                if isinstance(transient.public_timestamp, datetime):
                    public_timestamp = transient.public_timestamp.isoformat()
                trans_data = {
                    'name': transient.name,
                    'public_timestamp': public_timestamp,
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
        from datetime import datetime
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


def trigger_usage_log_roller():
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


def export_logs_to_influxdb():
    import json
    import os
    import gzip
    from influxdb_client import InfluxDBClient, Point
    from host.object_store import ObjectStore
    import tempfile
    # import sys
    from datetime import datetime

    DATA_INIT_S3_CONF = {
        'endpoint-url': 'https://rice1.osn.mghpcc.org',
        'region-name': 'osn',
        'aws_access_key_id': '',
        'aws_secret_access_key': '',
        'bucket': 'blast-astro-data',
    }

    s3 = ObjectStore(conf=DATA_INIT_S3_CONF)

    def download_data(object_key):
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            temp_file_path = fp.name
            fp.close()
            s3.download_object(
                path=object_key,
                file_path=temp_file_path)
            with gzip.open(temp_file_path) as archive_file:
                data = json.loads(archive_file.read())
        return data

    def construct_points(data):
        points = []
        for item in data:
            fields = item.get("fields")
            request_url = fields.get('request_url')
            request_method = fields.get('request_method')
            request_time = fields.get('request_time')
            submitted_data = fields.get('submitted_data')
            request_user = fields.get('request_user')
            request_ip = fields.get('request_ip')
            if not request_time or not request_url:
                print("Skipping invalid item:", item)
                continue

            point = Point('requests')
            point.field('request_url', request_url)
            point.field('request_method', request_method)
            point.field('submitted_data', submitted_data)
            point.field('request_user', request_user)
            point.field('request_ip', request_ip)
            point.time(request_time)
            points.append(point)
        return points

    def main():
        min_time = datetime.fromisoformat('2026-04-20T15:00:00')
        max_time = datetime.fromisoformat('2026-04-20T20:00:00')
        print(f'Analyzing logs starting from {min_time} ...')

        log_root = "/apps/blast/logs"
        objs = s3.client.list_objects(
            bucket_name=s3.bucket,
            prefix=log_root.strip('/'),
            recursive=True,
        )
        logs = []
        for object_key in [obj.object_name for obj in objs]:
            # print(object_key)
            log_date_str = object_key.split('.')[1]
            log_date_iso = (f'''{log_date_str[0:4]}-{log_date_str[4:6]}-{log_date_str[6:8]}T'''
                            f'''{log_date_str[8:10]}:{log_date_str[10:12]}:{log_date_str[12:14]}''')
            # print(log_date_iso)
            log_time = datetime.fromisoformat(log_date_iso)
            if log_time > min_time and log_time < max_time:
                print(f'''Downloading logs for {log_time}...''')
                data = download_data(object_key)
                logs.extend(data)
        # with open('/tmp/logs.json', 'w') as fp:
        #     json.dump(logs, fp, indent=2)

        points = construct_points(logs)

        token = os.getenv('INFLUXDB_INIT_ADMIN_TOKEN', '')
        org = os.getenv('INFLUXDB_INIT_ORG', 'blast')
        bucket = os.getenv('INFLUXDB_INIT_BUCKET', 'logs')
        url = os.getenv('INFLUXDB_URL', 'http://influxdb.blast.svc.cluster.local')

        client = InfluxDBClient(url=url, token=token, org=org)
        write_api = client.write_api()
        write_api.write(bucket=bucket, record=points)
        write_api.close()

    main()
