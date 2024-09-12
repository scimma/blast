#!/bin/env bash

set -o pipefail
set -e

# Create data folders on persistent volume and symlink to expected paths
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "${SCRIPT_DIR}"

bash initialize_data_dirs.sh

cd /tmp

extract_data_archive_file() {
  local file_path=$1
  local extract_dir=$2
  local original_dir=$(pwd)
  echo "INFO: Installing data from archive file \"${file_path}\"..."
  if [[ ! -f "${file_path}" ]]; then
    echo "ERROR: Data archive file \"${file_path}\" not found. Aborting."
    return 1
  fi
  echo "Extracting data archive..."
  # Data archive file has top-level directory "data"
  cd "${extract_dir}"
  tar --strip-components=1 -xzf "${DATA_ARCHIVE_FILE}"
  cd "${original_dir}"
}

verify_data_integrity() {
  # Verify data file integrity.
  local data_root_dir=$1
  local original_dir=$(pwd)
  cd "${data_root_dir}"
  echo "Verifying file integrity in data directory: $(pwd)"
  set +e
  md5sum --check --status "${SCRIPT_DIR}/blast-data.md5sums"
  DATA_INTEGRITY_VALID=$?
  set -e
  cd "${original_dir}"
  if [[ "${DATA_INTEGRITY_VALID}" == "0" ]]
  then
    return 0
  else
    echo "ERROR: Required data files fail integrity check."
    return 1
  fi
}

compare_checksum() {
  # This code borrows heavily from https://stackoverflow.com/a/61861691
  data_root_dir=$1
  prefix=$2
  file_base=$3
  file="$(readlink -f "${data_root_dir}")/${file_base}"
  s3_key="blast/blast-astro-data/${prefix}/${file_base}"
  partSizeInMb=16
  if [ ! -f "${file}" ]; then
      echo "ERROR: $file not found." 
      return 1;
  fi
  etag_remote=$(mc stat --json "${s3_key}" | jq --raw-output '.etag')
  fileSizeInMb=$(du -m "$file" | cut -f 1)
  parts=$((fileSizeInMb / partSizeInMb))
  if [[ $((fileSizeInMb % partSizeInMb)) -gt 0 ]]; then
      parts=$((parts + 1));
  fi
  checksumFile=$(mktemp -t s3md5.XXXXXXXXXXXXX)
  for (( part=0; part<$parts; part++ ))
  do
      skip=$((partSizeInMb * part))
      $(dd bs=1M count=$partSizeInMb skip=$skip if="$file" 2> /dev/null | md5sum >> $checksumFile)
  done
  etag=$(echo $(xxd -r -p $checksumFile | md5sum)-$parts | sed 's/ --/-/')
  rm $checksumFile
  if [[ "${etag_remote}" == "${etag}" ]]; then
    # echo "DEBUG: Required data file passed integrity check: ${file}"
    return 0
  else
    echo "ERROR: Comparing \"${file}\" to \"${s3_key}\"..."
    echo "ERROR: Calculated checksum: ${etag}"
    echo "ERROR: Source checksum:     ${etag_remote}"
    echo "ERROR: Required data file failed integrity check: ${file}"
    return 1
  fi
}

download_data_archive() {
  local data_root_dir=$1
  echo "INFO: Downloading data from archive..."
  mc alias set blast https://js2.jetstream-cloud.org:8001 anonymous
  # The trailing slashes are important!
  mc mirror --overwrite --json blast/blast-astro-data/v1/data/ "$(readlink -f "${data_root_dir}")/"
  declare -a file_paths=(
    "v2/data/sbipp/SBI_model.pt"
    "v2/data/sbipp/SBI_model_global.pt"
    "v2/data/sbipp/SBI_model_local.pt"
    "v2/data/sbipp_phot/sbi_phot_global.h5"
    "v2/data/sbipp_phot/sbi_phot_local.h5"
    "v3/data/sbi_training_sets/hatp_x_y_global.pkl"
    "v3/data/sbi_training_sets/hatp_x_y_local.pkl"
    "v3/data/sbi_training_sets/x_train_global.pkl"
    "v3/data/sbi_training_sets/x_train_local.pkl"
    "v3/data/sbi_training_sets/y_train_global.pkl"
    "v3/data/sbi_training_sets/y_train_local.pkl"
  )
  for file_path in ${file_paths[@]}
  do
    prefix="$(echo ${file_path} | sed -E 's#(v[0-9]+/data)\/.+#\1#')"
    data_file="$(echo ${file_path} | sed -E 's#(v[0-9]+/data)\/(.+)#\2#')"
    # echo "Comparing \"blast/blast-astro-data/${prefix}/${data_file}\" to \"$(readlink -f "${data_root_dir}")/${data_file}\"..."
    if ! compare_checksum "$(readlink -f "${data_root_dir}")" ${prefix} "${data_file}"
    then
      mc cp --json "blast/blast-astro-data/${prefix}/${data_file}" "$(readlink -f "${data_root_dir}")/${data_file}"
    fi
  done
}


# Verify data file integrity and attempt to (re)install required files if necessary
if ! verify_data_integrity "${DATA_ROOT_DIR}"
then
  # Download and install data from archive
  if [[ "${USE_LOCAL_ARCHIVE_FILE}" == "true" ]]
  then
    # Extract data from local archive file
    extract_data_archive_file "${DATA_ARCHIVE_FILE}" "${DATA_ROOT_DIR}"
  else
    # Download data from remote archive
    download_data_archive "${DATA_ROOT_DIR}"
  fi
  # Verify data file integrity
  if ! verify_data_integrity "${DATA_ROOT_DIR}"
  then
    echo "ERROR: Downloaded/extracted data files failed integrity check. Aborting."
    exit 1
  fi
  echo "Data installed."
fi

# Skip redundant installation of dustmap data and config file, where "init_data.py"
# executes "app/entrypoints/initialize_dustmaps.py", which downloads SFD files
# if they are missing and initializes a ".dustmapsrc" file.
# cd "${SCRIPT_DIR}"/..
# python init_data.py

echo "Data initialization complete."
