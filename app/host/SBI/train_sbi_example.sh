#!/bin/bash
#SBATCH -p dark
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --ntasks-per-core 1
#SBATCH --mem-per-cpu 10000
#SBATCH --time=72:00:00
#
# Usage (any arguments are passed straight to train_sbi.py):
#
#   Simulate photometry (default if no arguments are given):
#     sbatch train_sbi_example.sh --do-phot --zmin 0.0 --zmax 0.2 --outfile host/SBI/sbi_phot_1.h5
#
#   Ten photometry chunks in parallel, one file per array task:
#     sbatch --array=1-10 train_sbi_example.sh --do-phot --zmax 0.2
#     (with no --outfile, each task writes host/SBI/sbi_phot_${SLURM_ARRAY_TASK_ID}.h5)
#
#   Train on the chunks (quote wildcards so they resolve relative to app/):
#     sbatch train_sbi_example.sh --do-train --train-files 'host/SBI/sbi_phot_*.h5' \
#         --model-file host/SBI/SBI_model.pt --summary-file host/SBI/SBI_model_summary.p
#
#   Full option list:
#     python -m host.SBI.train_sbi --help

set -euo pipefail

set +u                            # conda's activate.d hooks use unset variables
eval "$(conda shell.bash hook)"   # makes `conda activate` work in a batch shell
conda activate blast
set -u
cd $BLAST_HOME_DIR/app

# Default output file; used when --outfile is not given on the command line.
export OUTFILE=${OUTFILE:-host/SBI/sbi_phot_${SLURM_ARRAY_TASK_ID:-1}.h5}

if [ "$#" -eq 0 ]; then
    set -- --do-phot --zmin 0.0 --zmax 0.2
fi

echo "Running: python -m host.SBI.train_sbi $*"
python -u -m host.SBI.train_sbi "$@"
