#!/bin/bash -l
#SBATCH --job-name=blast_transient_<transient_name>
#SBATCH --time=34:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=<outdir>/blast_transient_<transient_name>.log
#SBATCH --mem=12GB
#SBATCH -p dark

export BLAST_TRANSIENT_NAME=<transient_name>
export BLAST_TRANSIENT_RA=<transient_ra>
export BLAST_TRANSIENT_DEC=<transient_dec>

conda activate blast
cd <blastdir>
python manage.py runcrons host.slurm.run_single_transient.run_single --force
