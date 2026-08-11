from django.db import migrations

NEW_TASK_NAME = 'Host spectrum download'


def add_host_spectrum_task(apps, schema_editor):
    '''Add the new Host spectrum download task definition'''
    Task = apps.get_model("host", "Task")
    Status = apps.get_model("host", "Status")
    Task.objects.create(name=NEW_TASK_NAME)
    Status.objects.create(message='no host spectrum', type='error')


def add_host_spectrum_taskregisters(apps, schema_editor):
    '''Add task registers for the new task to existing transients so that they will run when retriggered.'''
    Transient = apps.get_model("host", "Transient")
    TaskRegister = apps.get_model("host", "TaskRegister")
    Task = apps.get_model("host", "Task")
    Status = apps.get_model("host", "Status")
    not_processed_status = Status.objects.get(message__exact="not processed")
    task = Task.objects.get(name=NEW_TASK_NAME)

    all_transients = Transient.objects.all()
    all_trs = TaskRegister.objects.select_related('transient').select_related('task').all()
    num_transients = len(all_transients)
    for idx, transient in enumerate(all_transients):
        print(f'[{idx + 1}/{num_transients}] Adding host spectrum task register for transient "{transient.name}"...')
        if not all_trs.filter(transient=transient, task=task):
            TaskRegister.objects.create(
                transient=transient,
                task=task,
                status=not_processed_status,
            )
        else:
            print(f'''WARNING: Unexpected TaskRegister object already exists for "{NEW_TASK_NAME}" '''
                  f'''on transient "{transient.name}".''')


class Migration(migrations.Migration):

    dependencies = [
        ('host', '0052_hostspectrum'),
    ]

    operations = [
        migrations.RunPython(add_host_spectrum_task),
        migrations.RunPython(add_host_spectrum_taskregisters),
    ]
