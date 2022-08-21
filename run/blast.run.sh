#!/bin/env bash
bash app/entrypoints/clean_data.sh
bash run/get_fsps_files.sh

case "$1" in
      test) docker compose --profile $1 --project-name blast -f docker/docker-compose.yml --env-file env/.env.dev up --build --exit-code-from app_test;;
        ci) docker compose --profile $1 --project-name blast -f docker/docker-compose.yml --env-file env/.env.ci up --build --exit-code-from app_ci;;
         *) docker compose --profile $1 --project-name blast -f docker/docker-compose.yml --env-file env/.env.dev up --build;;
esac
