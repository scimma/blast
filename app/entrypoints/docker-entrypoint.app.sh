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
  declare -a unit_tests=(
    "users.tests"
    "host.tests.test_aperture_construction"
    "host.tests.test_cutouts"
    "host.tests.test_ebv"
    "host.tests.test_host_match"
    "host.tests.test_models"
    "host.tests.test_photometry"
    "host.tests.test_processing"
    "host.tests.test_sedfitting"
    "host.tests.test_transient_name_server"
    "host.tests.test_transient_rename"
    "host.tests.test_utils"
    "host.tests.test_views"
    "api.tests.test_api.APITestDataset.test_dataset_get"
    "api.tests.test_api.APITestDataset.test_dataset_get_missing"
    "api.tests.test_api.APITestDataset.test_dataset_delete"
    "api.tests.test_api.APITestDataset.test_dataset_delete_with_files"
    "api.tests.test_api.APITestAlias.test_alias"
  )
  for unit_test in ${unit_tests[@]}; do
    echo "Running \"${unit_test}\"..."
    python manage.py test --exclude-tag=download "${unit_test}"
  done
  exit 0
fi


# Start server
if [[ $DEV_MODE == 1 ]]; then
  python manage.py --exclude-tag=download runserver 0.0.0.0:${WEB_APP_PORT}
else
  bash entrypoints/wait-for-it.sh ${WEB_SERVER_HOST}:${WEB_SERVER_PORT} --timeout=0
  gunicorn app.wsgi --timeout 0 --bind 0.0.0.0:${WEB_APP_PORT} --workers=${GUNICORN_WORKERS:=1}
fi
