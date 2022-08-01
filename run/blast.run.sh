#!/bin/env bash
bash app/entrypoints/clean_data.sh

case "$1" in
      test) docker compose --profile $1 --project-name blast -f docker/docker-compose.yml --env-file env/.env.dev up --build --exit-code-from app_test;;
        ci) docker compose --profile $1 --project-name blast -f docker/docker-compose.yml --env-file env/.env.ci up --build --exit-code-from app_ci;;
      old_slim_dev) docker compose --project-name blast -f docker/docker-compose.old.slim_dev.yml --env-file env/.env.dev up --build;;
      old_full_dev) docker compose --project-name blast -f docker/docker-compose.old.full_dev.yml --env-file env/.env.dev up --build;;
      old_test) docker compose --project-name blast -f docker/docker-compose.old.test.yml --env-file env/.env.dev up --build --exit-code-from app_test;;
      old_docs) docker compose --project-name blast -f docker/docker-compose.old.docs.yml up --build;;
         *) docker compose --profile $1 --project-name blast -f docker/docker-compose.yml --env-file env/.env.dev up --build;;
esac
