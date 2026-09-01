#!/bin/env bash

set -eo pipefail

bash entrypoints/install_dustmaps_config.sh

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
until mc alias set object-storage ${S3_ENDPOINT_URL} "" "" && mc ping -c 1 object-storage; do
  echo "Waiting for object storage..."
  sleep 5
done
echo "Running astro data initialization script..."
# Create data folders on persistent volume and symlink to expected paths
bash entrypoints/initialize_data_dirs.sh
# Verify and download missing and invalid files
if [[ "${SKIP_INITIALIZATION}" != "true" ]]
then
  python entrypoints/initialize_data.py
else
  echo "Skipping data initialization."
fi
echo "Data initialization complete."

## Initialize Django database and static files
##
bash entrypoints/wait-for-it.sh ${DB_HOST}:${DB_PORT} --timeout=0

echo "Running database initialization script..."
python init_app.py
echo "Django database initialization complete."

# Create semaphore file checked by the Docker Compose healthcheck
touch /tmp/celery-beat-ready

if [[ $DISABLE_CELERY_BEAT == "true" ]]; then
    echo "Celery Beat is disabled. Suspending."
    sleep infinity
fi

bash entrypoints/wait-for-it.sh ${MESSAGE_BROKER_HOST}:${MESSAGE_BROKER_PORT} --timeout=0

if [[ $DEV_MODE == 1 ]]; then
  watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- \
  celery -A app beat -l ${CELERY_LOG_LEVEL:-DEBUG}
else
  celery -A app beat -l ${CELERY_LOG_LEVEL:-INFO}
fi
