#!/bin/bash -x

#SBATCH --output=chembl-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=chembl
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=yl6624@nyu.edu
#SBATCH --account=pr_136_tandon_advanced

model_types=("arctic-zs" "roberta-zs" "mpnet-zs")
serializations=("header_values_default" "header_values_prefix")
usecases=("joinable" "semjoinable" "unionable" "viewunion")
llm_model="gpt-4o-mini"

for usecase in "${usecases[@]}"; do
    for model_type in "${model_types[@]}"; do
        for serialization in "${serializations[@]}"; do
            echo "Running model type: $model_type with serialization: $serialization for usecase: $usecase"
            python rema-sm/retrieve_match.py \
                --model_type "$model_type" \
                --serialization "$serialization" \
                --dataset "chembl-$usecase" \
                --cand_k 20 \
                --llm_model "$llm_model"
        done
    done
done