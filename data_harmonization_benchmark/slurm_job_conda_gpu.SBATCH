#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem 64GB
#SBATCH --time 24:00:00
#SBATCH --mail-user=yfw215@nyu.edu
#SBATCH --account=pr_136_tandon_advanced

module purge;
module load anaconda3/2020.07;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda create -p ./penv_${1} python=3.9;
conda activate ./penv_${1};
export PATH=./penv_${1}/bin:$PATH;
conda install cuda -c nvidia;

pip install -r requirements.txt
pip install -r matchers/${1}/requirements.txt
python3 runbenchmark.py -m ${1} -c datasets/${2} -n ${3} -k ${4}