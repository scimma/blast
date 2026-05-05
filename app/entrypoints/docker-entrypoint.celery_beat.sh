#!/bin/env bash

set -e

bash entrypoints/install_dustmaps_config.sh

if [[ $DISABLE_CELERY_BEAT == "true" ]]; then
    echo "Celery Beat is disabled. Suspending."
    sleep infinity
fi

bash entrypoints/wait-for-it.sh ${DB_HOST}:${DB_PORT} --timeout=0
bash entrypoints/wait-for-it.sh ${MESSAGE_BROKER_HOST}:${MESSAGE_BROKER_PORT} --timeout=0
bash entrypoints/wait-for-it.sh ${WEB_APP_HOST}:${WEB_APP_PORT} --timeout=0

bash entrypoints/initialize_data_dirs.sh

if [[ $DEV_MODE == 1 ]]; then
  watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- \
  celery -A app beat -l ${CELERY_LOG_LEVEL:-DEBUG}
else
  celery -A app beat -l ${CELERY_LOG_LEVEL:-INFO}
fi
