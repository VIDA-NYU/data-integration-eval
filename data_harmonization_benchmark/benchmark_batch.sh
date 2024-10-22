#!/bin/bash

usecases="ChEMBL MagellanHumanCurated OpenData TPC-DI WikidataHumanCurated"
# usecases="amazon_google_exp beeradvo_ratebeer fodors_zagats itunes_amazon walmart_amazon dblp_acm dblp_scholar Musicians_joinable Musicians_semjoinable Musicians_unionable Musicians_viewunion"
# matchers="coma similarity_flooding cupid distribution_based jaccard_distance two_phase"
matchers="Unicorn"
n_runs=3
top_k=20
use_gpu=true

rm -rf tmp/logs/*
for usecase in $usecases
do
    for matcher in $matchers
    do  
        if [[ $use_gpu = true ]]; then
            echo "[GPU] Run benchmark for ${matcher} on usecase:${usecase}"
            sbatch --output tmp/logs/benchmark_job_${matcher}_${usecase}.out slurm_job_conda_gpu.SBATCH $matcher $usecase $n_runs $top_k
        else
            echo "[CPU] Run benchmark for ${matcher} on usecase:${usecase}"
            sbatch --output tmp/logs/benchmark_job_${matcher}_${usecase}.out slurm_job.SBATCH $matcher $usecase $n_runs $top_k
        fi
    done
done