#!/bin/env bash

bash entrypoints/wait-for-it.sh ${DB_HOST}:${DB_PORT} --timeout=0 &&
bash entrypoints/wait-for-it.sh ${WEB_SERVER_HOST}:${WEB_SERVER_PORT} --timeout=0 &&
python manage.py makemigrations &&
python manage.py migrate &&
gunicorn app.wsgi:application --bind 0.0.0.0:${WEB_APP_PORT}
