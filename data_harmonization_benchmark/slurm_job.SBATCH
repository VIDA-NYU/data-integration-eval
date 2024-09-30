#!/bin/bash
#SBATCH -c 1
#SBATCH --mem 32GB
#SBATCH --time 12:00:00
#SBATCH --mail-user=yfw215@nyu.edu
#SBATCH --account=pr_136_tandon_advanced

singularity exec --overlay overlay-50G-10M.ext3:ro /scratch/work/public/singularity/ubuntu-20.04.4.sif /bin/bash -c "source /ext3/env.sh; python3 runbenchmark.py -m ${1} -c datasets/${2} -n ${3}"