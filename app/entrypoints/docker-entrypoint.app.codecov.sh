#!/bin/env bash
bash entrypoints/wait-for-it.sh database:${DB_PORT} --timeout=0 &&
python manage.py makemigrations &&
python manage.py migrate &&
pip install coverage &&
coverage run manage.py test host.tests -v 2 &&
