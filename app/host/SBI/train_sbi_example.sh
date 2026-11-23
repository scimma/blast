#!/bin/bash
#SBATCH -p dark
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --ntasks-per-core 1
#SBATCH --mem-per-cpu 10000
#SBATCH --time=72:00:00
#
# Usage (any arguments are passed straight to train_sbi.py). Tag file names
# with the redshift range so runs with different settings never collide:
#
#   1. Simulate photometry, ten chunks in parallel (one file per array task):
#        sbatch --array=1-10 train_sbi_example.sh --do-phot --zmin 0.0 --zmax 0.2 \
#            --outfile 'host/SBI/sbi_phot_z0.0-0.2_${SLURM_ARRAY_TASK_ID}.h5'
#      (with no --outfile, each task writes host/SBI/sbi_phot_${SLURM_ARRAY_TASK_ID}.h5)
#
#   2a. Train the spec-z model (redshift is an input):
#        sbatch train_sbi_example.sh --do-train \
#            --train-files 'host/SBI/sbi_phot_z0.0-0.2_*.h5' \
#            --merged-file  host/SBI/sbi_phot_z0.0-0.2.h5 \
#            --model-file   host/SBI/SBI_model_z0.0-0.2.pt \
#            --summary-file host/SBI/SBI_model_z0.0-0.2_summary.p
#
#   2b. Train the photo-z model on the same simulations (redshift is inferred):
#        sbatch train_sbi_example.sh --do-train --do-zphot-model \
#            --train-files 'host/SBI/sbi_phot_z0.0-0.2_*.h5' \
#            --merged-file  '' \
#            --model-file   host/SBI/SBI_model_zphot_z0.0-0.2.pt \
#            --summary-file host/SBI/SBI_model_zphot_z0.0-0.2_summary.p
#      (without explicit paths it writes SBI_model_zphot.pt / SBI_model_zphot_summary.p)
#
#   Quote wildcards and ${SLURM_ARRAY_TASK_ID} in single quotes so they are
#   resolved inside the job (relative to app/), not at submit time.
#
#   Full option list:
#     python -m host.SBI.train_sbi --help

set -euo pipefail

set +u                            # conda's activate.d hooks use unset variables
eval "$(conda shell.bash hook)"   # makes `conda activate` work in a batch shell
conda activate blast
set -u
cd "${BLAST_HOME_DIR:?set BLAST_HOME_DIR to the blast checkout before running sbatch}/app"

# Default output file; used when --outfile is not given on the command line.
export OUTFILE=${OUTFILE:-host/SBI/sbi_phot_${SLURM_ARRAY_TASK_ID:-1}.h5}

if [ "$#" -eq 0 ]; then
    set -- --do-phot --zmin 0.0 --zmax 0.2
fi

# Expand ${SLURM_ARRAY_TASK_ID} inside single-quoted arguments (e.g. --outfile)
args=()
for a in "$@"; do
    args+=("${a//\$\{SLURM_ARRAY_TASK_ID\}/${SLURM_ARRAY_TASK_ID:-1}}")
done

echo "Running: python -m host.SBI.train_sbi ${args[*]}"
python -u -m host.SBI.train_sbi "${args[@]}"
