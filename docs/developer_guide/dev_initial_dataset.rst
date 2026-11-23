Initial Data File Set
=====================

Blast needs a specific set of initial data files to be installed. This file set evolves
with the software, and such it must be version controlled. Due to the large size of this
file set, we host it on our S3-compatible object store in a bucket with versioning 
enabled.

The explicit manifest of files and their relevant metadata is stored in 
:code:`app/entrypoints/blast-data.json`. This manifest is parsed during the initialization
routine and used to verify the integrity of each required file. This enables the process to
be very efficient without sacrificing rigorous version control; only the files that are 
missing or corrupt are downloaded.

When this initial file set is modified, for example to revise the SBI training model files, 
several steps must be completed:

1. Each new or modified file must be uploaded to the S3 bucket. If the file already exists,
a new object version will be automatically generated. You can check this using an S3 client 
such as the MinIO :code:`mc` CLI as in the example below.

   .. code ::
   
       $ mc ls --versions js-blast/blast-astro-data/init/data/sbi_training_sets/hatp_x_y_global.pkl
   
       [2024-09-11 16:11:29 CDT]  62MiB STANDARD t7be7U7pCQp1bUmi9dwsErXyMSLGThk v2 PUT hatp_x_y_global.pkl
       [2024-09-11 15:32:46 CDT]  62MiB STANDARD 3Pd7n-9ywWidv9vQBtjkEMFj6TPDDvK v1 PUT hatp_x_y_global.pkl
   
2. To rebuild the file manifest, run the following:

   .. code :: python
   
       python app/entrypoints/initialize_data.py manifest
