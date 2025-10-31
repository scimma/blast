#!/bin/env bash
bash entrypoints/wait-for-it.sh ${DB_HOST}:${DB_PORT} --timeout=0 &&
bash entrypoints/wait-for-it.sh ${MESSAGE_BROKER_HOST}:${MESSAGE_BROKER_PORT} --timeout=0 &&
bash entrypoints/wait-for-it.sh ${WEB_APP_HOST}:${WEB_APP_PORT} --timeout=0 &&
celery --broker=amqp://${RABBITMQ_USERNAME}:${RABBITMQ_PASSWORD}@${MESSAGE_BROKER_HOST}:${MESSAGE_BROKER_PORT}// flower --port=${FLOWER_PORT:-8888} --url_prefix=${FLOWER_URL_PREFIX:-}
