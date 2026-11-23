Training SBI
============

Training new SBI models is done asynchronously from the Blast web application.
Currently, Blast trains three models in different redshift regimes: z < 0.015,
0.015 < z < 0.2, and 0.2 < z < 1.0.  This is because limited training samples
can result in large uncertainties, particularly in regimes where small increments
in redshift yield large changes in magnitude.

Training is performed by running the :code:`train_sbi.py` script in the :code:`app/host/SBI`
subdirectory.  Each redshift range will need to be either trained individually, or you will need
to edit the example below to avoid file naming collisions.

First, run the simulations in 10 batches of 150000 each:

   .. code ::

       sbatch --array=1-10 train_sbi_example.sh --do-phot --zmin 0.015 --zmax 0.2     # writes sbi_phot_1..10.h5

Once these have finished, train the model using those files:

   .. code ::

      sbatch train_sbi_example.sh --do-train --train-files 'host/SBI/sbi_phot_*.h5' --model-file host/SBI/SBI_model_0015_0200.pt

The SBI example will need to be adapted to your cluster environment.  It assumes you have a conda environment
called "blast" and an environment variable $BLAST_HOME_DIR for your top-level blast code directory.  The details
of the slurm environment/partitions/nodes will also need to be adjusted.

To run the photo-z model, use the :code:`--do-zphot-model` argument.

Once the new model has been trained, it will need to be added to the S3 bucket for use in Blast.
