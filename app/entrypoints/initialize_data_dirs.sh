#!/bin/bash
set -e

# Ensure data root directory is configured
if [[ "${DATA_ROOT_DIR}x" == "x" ]]; then
  echo "ERROR: DATA_ROOT_DIR environment variable must not be empty. Aborting."
  exit 1
fi

cd "${DATA_ROOT_DIR}"

# The creation of symlinks should error if there is a non-symlink
# file or folder where the symlink should be.
if [[ ! -L "${CUTOUT_ROOT}" ]]; then
  mkdir -p "${DATA_ROOT_DIR}"/cutout_cdn
  mkdir -p "$(dirname "${CUTOUT_ROOT}")"
  ln -s "${DATA_ROOT_DIR}/cutout_cdn" "${CUTOUT_ROOT}"
fi
# if [[ ! -L "${CUTOUT_SCRATCH_ROOT}" ]]; then
#   mkdir -p "${DATA_ROOT_DIR}"/cutout_scratch
#   mkdir -p "$(dirname "${CUTOUT_SCRATCH_ROOT}")"
#   ln -s "${DATA_ROOT_DIR}/cutout_scratch" "${CUTOUT_SCRATCH_ROOT}"
# fi
if [[ ! -L "${SED_OUTPUT_ROOT}" ]]; then
  mkdir -p "${DATA_ROOT_DIR}"/sed_output
  mkdir -p "$(dirname "${SED_OUTPUT_ROOT}")"
  ln -s "${DATA_ROOT_DIR}/sed_output" "${SED_OUTPUT_ROOT}"
fi
if [[ ! -L "${GHOST_DATA_ROOT}" ]]; then
  mkdir -p "${DATA_ROOT_DIR}"/ghost_data
  mkdir -p "$(dirname "${GHOST_DATA_ROOT}")"
  ln -s "${DATA_ROOT_DIR}/ghost_data" "${GHOST_DATA_ROOT}"
fi
# if [[ ! -L "${GHOST_OUTPUT_ROOT}" ]]; then
#   mkdir -p "${DATA_ROOT_DIR}"/ghost_output
#   mkdir -p "$(dirname "${GHOST_OUTPUT_ROOT}")"
#   ln -s "${DATA_ROOT_DIR}/ghost_output" "${GHOST_OUTPUT_ROOT}"
# fi
if [[ ! -L "${TNS_STAGING_ROOT}" ]]; then
  mkdir -p "${DATA_ROOT_DIR}"/tns_staging
  mkdir -p "$(dirname "${TNS_STAGING_ROOT}")"
  ln -s "${DATA_ROOT_DIR}/tns_staging" "${TNS_STAGING_ROOT}"
fi
if [[ ! -L "${TRANSMISSION_CURVES_ROOT}" ]]; then
  mkdir -p "${DATA_ROOT_DIR}"/transmission
  mkdir -p "$(dirname "${TRANSMISSION_CURVES_ROOT}")"
  ln -s "${DATA_ROOT_DIR}/transmission" "${TRANSMISSION_CURVES_ROOT}"
fi
if [[ ! -L "${DUSTMAPS_DATA_ROOT}" ]]; then
  mkdir -p "${DATA_ROOT_DIR}"/dustmaps
  mkdir -p "$(dirname "${DUSTMAPS_DATA_ROOT}")"
  ln -s "${DATA_ROOT_DIR}/dustmaps" "${DUSTMAPS_DATA_ROOT}"
fi
if [[ ! -L "${SPS_HOME}" ]]; then
  mkdir -p "${DATA_ROOT_DIR}"/fsps
  mkdir -p "$(dirname "${SPS_HOME}")"
  ln -s "${DATA_ROOT_DIR}/fsps" "${SPS_HOME}"
fi
if [[ ! -L "${SBIPP_ROOT}" ]]; then
  mkdir -p "${DATA_ROOT_DIR}"/sbipp
  mkdir -p "$(dirname "${SBIPP_ROOT}")"
  ln -s "${DATA_ROOT_DIR}/sbipp" "${SBIPP_ROOT}"
fi
if [[ ! -L "${SBIPP_PHOT_ROOT}" ]]; then
  mkdir -p "${DATA_ROOT_DIR}"/sbipp_phot
  mkdir -p "$(dirname "${SBIPP_PHOT_ROOT}")"
  ln -s "${DATA_ROOT_DIR}/sbipp_phot" "${SBIPP_PHOT_ROOT}"
fi
if [[ ! -L "${SBI_TRAINING_ROOT}" ]]; then
  mkdir -p "${DATA_ROOT_DIR}"/sbi_training_sets
  mkdir -p "$(dirname "${SBI_TRAINING_ROOT}")"
  ln -s "${DATA_ROOT_DIR}/sbi_training_sets" "${SBI_TRAINING_ROOT}"
fi
