#!/bin/env bash
set -e

PROFILE=$1
PURGE_OPTION=$2

ENV_FILE="env/.env.dev"
TARGET_SERVICE="app_dev"
COMPOSE_ARGS=""
case "$PROFILE" in
  "")
    echo "ERROR: You must specify a profile (e.g. $0 slim_dev)"
    exit 1
    ;;
  ci)
    TARGET_SERVICE="app_ci"
    ENV_FILE="env/.env.ci"
    COMPOSE_ARGS="--build --exit-code-from app_ci"
    ;;
  slim_dev)
    COMPOSE_ARGS="--build --abort-on-container-exit"
    ;;
  slim_prod | full_prod)
    TARGET_SERVICE="app"
    ;;
  *)
    COMPOSE_ARGS="--build"
    ;;
esac

PURGE_VOLUMES=""
case "${PURGE_OPTION}" in
  "")
    echo "Retaining all data volumes."
    ;;
  "--purge-all")
    COMPOSE_ARGS="--volumes"
    echo "Purging all data volumes..."
  ;;
  "--purge-db")
    COMPOSE_ARGS=""
    PURGE_VOLUMES="${COMPOSE_PROJECT_NAME:-blast}_blast-db ${COMPOSE_PROJECT_NAME:-blast}_django-static"
    echo "Purging Django database and static file volumes..."
  ;;
  "--purge-data")
    COMPOSE_ARGS=""
    PURGE_VOLUMES="${COMPOSE_PROJECT_NAME:-blast}_blast-data"
    echo "Purging astro data volume..."
  ;;
  *)
    echo "ERROR: Invalid purge option."
    exit 1
    ;;
esac

COMPOSE_CONFIG=" --profile $PROFILE"
COMPOSE_CONFIG="${COMPOSE_CONFIG} --project-name ${COMPOSE_PROJECT_NAME:-blast}"
if [[ $PROFILE == "docs" ]]; then
  COMPOSE_CONFIG="${COMPOSE_CONFIG} -f docker/docker-compose.docs.yaml"
else
  COMPOSE_CONFIG="${COMPOSE_CONFIG} -f docker/docker-compose.yml"
  COMPOSE_CONFIG="${COMPOSE_CONFIG} --env-file env/.env.default"
  COMPOSE_CONFIG="${COMPOSE_CONFIG} --env-file ${ENV_FILE}"
fi

export COMPOSE_CONFIG
export COMPOSE_ARGS
export PURGE_VOLUMES
export TARGET_SERVICE
