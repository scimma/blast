#!/bin/env bash

bash entrypoints/wait-for-it.sh ${DB_HOST}:${DB_PORT} --timeout=0 &&
bash entrypoints/wait-for-it.sh ${MESSAGE_BROKER_HOST}:${MESSAGE_BROKER_PORT} --timeout=0 &&
bash entrypoints/wait-for-it.sh ${WEB_APP_HOST}:${WEB_APP_PORT} --timeout=0 &&
celery -A app flower --port=${FLOWER_PORT:-8888} --url_prefix=${FLOWER_URL_PREFIX:-}
