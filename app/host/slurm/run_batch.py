#!/usr/bin/env python
import argparse
import glob
import os
import time

import astropy.table as at


def main(
    batchfile,
    batchdir,
    blastdir="/lustre/hpc/storage/dark/YSE/blast/app",
#/n/holystore01/LABS/berger_lab/Lab/djones01/blast/app",
):
    data = at.Table.read(batchfile)

    if not os.path.exists(batchdir):
        os.makedirs(batchdir)

    for d in data:
        sedfiles = glob.glob(f"../../../data/sed_output/{d['transient_name']}/*global*")
        if len(sedfiles):
            os.system(f"rm {sedfiles[0]}")
            # continue
        slurmfile = f"{batchdir}/slurm_{d['transient_name']}.sh"

        with open("slurm_template.sh", "r") as fin, open(slurmfile, "w") as fout:
            for line in fin:
                line = (
                    line.replace("\n", "")
                    .replace("<transient_name>", str(d["transient_name"]))
                    .replace("<transient_ra>", str(d["transient_ra"]))
                    .replace("<transient_dec>", str(d["transient_dec"]))
                    .replace("<outdir>", batchdir)
                    .replace("<blastdir>", blastdir)
                )
                if "transient_redshift" in d.keys():
                    line = line.replace(
                        "<transient_redshift>", str(d["transient_redshift"])
                    )
                else:
                    line = line.replace(
                        "export BLAST_TRANSIENT_REDSHIFT=<transient_redshift>", ""
                    )
                print(line, file=fout)
        os.system(f"sbatch {slurmfile}")
        time.sleep(60 * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="batch submission for slurm systems")
    parser.add_argument("batchfile", help="batch CSV with transient name,ra,dec")
    parser.add_argument("batchdir", help="output directory for logs, job scripts")
    args = vars(parser.parse_args())

    main(args["batchfile"], args["batchdir"])
