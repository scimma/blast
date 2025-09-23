"""
Defines the natural keys for model objects to be de-serialized with.
"""
from django.db import models
from datetime import datetime, timedelta, timezone
from django.conf import settings
from host.log import get_logger
logger = get_logger(__name__)



class TransientManager(models.Manager):
    def get_by_natural_key(self, name):
        return self.get(name=name)


class StatusManager(models.Manager):
    def get_by_natural_key(self, message):
        return self.get(message=message)


class TaskManager(models.Manager):
    def get_by_natural_key(self, name):
        return self.get(name=name)


class SurveyManager(models.Manager):
    def get_by_natural_key(self, name):
        return self.get(name=name)


class CatalogManager(models.Manager):
    def get_by_natural_key(self, name):
        return self.get(name=name)


class FilterManager(models.Manager):
    def get_by_natural_key(self, name):
        return self.get(name=name)


class HostManager(models.Manager):
    def get_by_natural_key(self, name):
        return self.get(name=name)


class CutoutManager(models.Manager):
    def get_by_natural_key(self, name):
        return self.get(name=name)


class ApertureManager(models.Manager):
    def get_by_natural_key(self, name):
        return self.get(name=name)


class TaskLockManager(models.Manager):
    '''This mutex logic assumes cooperation between the processes using the lock.
    There is no access control to prevent concurrent processes from releasing an
    existing lock before the lock owner completes its execution.'''
    def new_expiration_time(self, name):
        '''Calculate the expiration time for a new lock, which can vary depending on the lock purpose.
        Expiration timeout values must be in units of seconds.'''
        if name in ['tns_query', 'ned_query']:
            expiration_period = settings.QUERY_TIMEOUT
        else:
            expiration_period = 60
        time_threshold = datetime.now(timezone.utc) + timedelta(seconds=expiration_period)
        return time_threshold

    def is_expired(self, name):
        lock_query = self.filter(name=name)
        if not lock_query:
            return True
        lock = lock_query[0]
        logger.debug(f'''now: {datetime.now(timezone.utc)}, expires: {lock.time_expires}''')
        return datetime.now(timezone.utc) > lock.time_expires

    def is_locked(self, name):
        if self.filter(name=name):
            return True
        return False

    def release_lock(self, name):
        # Attempt to delete the lock
        lock_query = self.filter(name=name)
        if lock_query:
            lock_query.delete()

    def create_lock(self, name):
        lock = self.create(name=name, time_expires=self.new_expiration_time(name))
        logger.debug(f'''New lock created: {lock}''')
        return lock

    def request_lock(self, name):
        lock_query = self.filter(name=name)
        # Create a lock if there none exists
        if not lock_query:
            self.create_lock(name)
            return True
        else:
            lock = lock_query[0]
            # Release a lock that is older than the expiration period and create a new one.
            if datetime.now(timezone.utc) > lock.time_expires:
                logger.debug(f'''Releasing expired lock "{name}"...''')
                lock_query.delete()
                self.create_lock(name)
                return True
        # Otherwise, deny the request.
        return False
