#!/bin/bash

set -eo pipefail

# If .dustmapsrc has been mounted as a read-only file at /tmp/.dustmapsrc,
# copy it to the expected location /root/.dustmapsrc if it does not already exist.
if [[ ! -f "/root/.dustmapsrc" && -f "/tmp/.dustmapsrc" ]]; then
  cp /tmp/.dustmapsrc /root/.dustmapsrc
fi

INIT_STARTED_DATA="${DATA_ROOT_DIR}/.initializing_data"
INIT_STARTED_DB="${DATA_ROOT_DIR}/.initializing_db"

if [[ "${FORCE_INITIALIZATION}" == "true" ]]; then
  rm -f "${INIT_STARTED_DATA}"
  rm -f "${INIT_STARTED_DB}"
fi

# Migrations should be created manually by developers and committed with the source code repo.
# Set the MAKE_MIGRATIONS env var to a non-empty string to create migration scripts
# after changes are made to the Django ORM models.
if [ -n "$MAKE_MIGRATIONS" ]; then
  echo "Generating database migration scripts..."
  python manage.py makemigrations --no-input
  exit 0
fi

## Initialize astro data
##

if [[ -f "${INIT_STARTED_DATA}" ]]
then
  echo "Astro data is currently being initialized (\"${INIT_STARTED_DATA}\" exists)."
  sleep 10
  exit 1
else
  echo "\"${INIT_STARTED_DATA}\" not found. Running initialization script..."
  touch "${INIT_STARTED_DATA}"

  # Create data folders on persistent volume and symlink to expected paths
  bash entrypoints/initialize_data_dirs.sh
  # Verify and download missing and invalid files
  python entrypoints/initialize_data.py

  rm -f "${INIT_STARTED_DATA}"
  echo "Data initialization complete."
fi

## Initialize Django database and static files
##
bash entrypoints/wait-for-it.sh ${DATABASE_HOST}:${DATABASE_PORT} --timeout=0

if [[ -f "${INIT_STARTED_DB}" ]]
then
  echo "Django database and static files are currently being initialized (\"${INIT_STARTED_DB}\" exists)."
  sleep 10
  exit 1
else
  echo "\"${INIT_STARTED_DB}\" not found. Running initialization script..."
  touch "${INIT_STARTED_DB}"

  python init_app.py

  rm -f "${INIT_STARTED_DB}"
  echo "Django database initialization complete."
fi

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
