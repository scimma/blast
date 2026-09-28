#!/bin/bash

set -eo pipefail

bash entrypoints/install_dustmaps_config.sh

# Create data folders on persistent volume and symlink to expected paths
bash entrypoints/initialize_data_dirs.sh

bash entrypoints/wait-for-it.sh ${DB_HOST}:${DB_PORT} --timeout=0

# If test mode, run tests and exit
if [[ $TEST_MODE == 1 ]]; then
  set -e
  # TODO: Restore the use of the "coverage" tool in a way that preserves the independent
  #       database instances associated with each test run.
  python manage.py test users.tests
  python manage.py test host.tests.test_aperture_construction
  python manage.py test host.tests.test_cutouts
  python manage.py test host.tests.test_ebv
  python manage.py test host.tests.test_host_match
  python manage.py test host.tests.test_models
  python manage.py test host.tests.test_panstarrs
  python manage.py test host.tests.test_photometry
  python manage.py test host.tests.test_processing
  python manage.py test host.tests.test_sedfitting
  python manage.py test host.tests.test_transient_name_server
  python manage.py test host.tests.test_transient_rename
  python manage.py test host.tests.test_utils
  python manage.py test host.tests.test_views
  python manage.py test api.tests.test_api.APITestDataset.test_dataset_get
  python manage.py test api.tests.test_api.APITestDataset.test_dataset_get_missing
  python manage.py test api.tests.test_api.APITestDataset.test_dataset_delete
  python manage.py test api.tests.test_api.APITestDataset.test_dataset_delete_with_files
  python manage.py test api.tests.test_api.APITestAlias.test_alias
  exit 0
fi


# Start server
if [[ $DEV_MODE == 1 ]]; then
  python manage.py runserver 0.0.0.0:${WEB_APP_PORT}
else
  bash entrypoints/wait-for-it.sh ${WEB_SERVER_HOST}:${WEB_SERVER_PORT} --timeout=0
  gunicorn app.wsgi --timeout 0 --bind 0.0.0.0:${WEB_APP_PORT} --workers=${GUNICORN_WORKERS:=1}
fi
