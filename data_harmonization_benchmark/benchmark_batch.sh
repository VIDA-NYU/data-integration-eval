#!/bin/bash

# usecases="cao clark dou huang krug satpathy vasaikar wang"
usecases="amazon_google_exp beeradvo_ratebeer fodors_zagats"
# matchers="coma similarity_flooding cupid distribution_based jaccard_distance two_phase"
matchers="two_phase coma"
n_runs=3

rm -rf tmp/logs/*
for usecase in $usecases
do
    for matcher in $matchers
    do
        echo "Run benchmark for ${matcher} on usecase:${usecase}"
        sbatch --output tmp/logs/benchmark_job_${matcher}_${usecase}.out slurm_job.SBATCH $matcher $usecase $n_runs
    done
done