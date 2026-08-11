#!/bin/bash

set -eo pipefail

bash entrypoints/install_dustmaps_config.sh

bash entrypoints/wait-for-it.sh ${DB_HOST}:${DB_PORT} --timeout=0
bash entrypoints/wait-for-it.sh ${MESSAGE_BROKER_HOST}:${MESSAGE_BROKER_PORT} --timeout=0

# If test mode, run tests and exit
if [[ $TEST_MODE == 1 ]]; then
  set -e
  coverage run manage.py test \
    --exclude-tag=download \
    host.tests api.tests users.tests \
    -v 2
  coverage report -i --omit=host/tests/*,host/migrations/*,app/*,host/urls.py,host/admin.py,host/apps.py,host/__init__.py,manage.py
  coverage xml -i
  exit 0
fi


# Start server
if [[ $DEV_MODE == 1 ]]; then
  python manage.py runserver 0.0.0.0:${WEB_APP_PORT}
else
  bash entrypoints/wait-for-it.sh ${WEB_SERVER_HOST}:${WEB_SERVER_PORT} --timeout=0
  gunicorn app.wsgi --timeout 0 --bind 0.0.0.0:${WEB_APP_PORT} --workers=${GUNICORN_WORKERS:=1}
fi
