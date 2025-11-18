#!/bin/env bash

set -e

if [[ $DISABLE_CELERY_BEAT == "true" ]]; then
    echo "Celery Beat is disabled. Terminating."
    exit 0
fi

bash entrypoints/wait-for-it.sh ${DB_HOST}:${DB_PORT} --timeout=0
bash entrypoints/wait-for-it.sh ${MESSAGE_BROKER_HOST}:${MESSAGE_BROKER_PORT} --timeout=0
bash entrypoints/wait-for-it.sh ${WEB_APP_HOST}:${WEB_APP_PORT} --timeout=0

# If .dustmapsrc has been mounted as a read-only file at /tmp/.dustmapsrc,
# copy it to the expected location /root/.dustmapsrc if it does not already exist.
if [[ ! -f "/root/.dustmapsrc" && -f "/tmp/.dustmapsrc" ]]; then
  cp /tmp/.dustmapsrc /root/.dustmapsrc
fi

bash entrypoints/initialize_data_dirs.sh

if [[ $DEV_MODE == 1 ]]; then
  watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- \
  celery -A app beat -l DEBUG
else
  celery -A app beat -l INFO
fi
