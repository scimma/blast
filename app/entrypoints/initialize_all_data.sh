#!/bin/env bash

set -o pipefail
set -e

# Create data folders on persistent volume and symlink to expected paths
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "${SCRIPT_DIR}"

# TODO: S3: Initialize buckets instead of local directories
bash initialize_data_dirs.sh

# Verify and download missing and invalid files
python initialize_data.py download

echo "Data initialization complete."
