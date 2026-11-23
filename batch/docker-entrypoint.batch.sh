#!/bin/env bash

set -eo pipefail

bash /wait-for-it.sh ${WEB_APP_HOST}:${WEB_APP_PORT} --timeout=0
python3 /run_batch.py /input.csv
