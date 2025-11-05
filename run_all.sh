#!/bin/bash

models=("tcformer" "atcnet" "eegnet" "shallownet" "basenet" "eegtcnet" "eegconformer" "tsseffnet" "ctnet" "mscformer" "ratiowavenet")
datasets=("bcic2a" "bcic2b" "hgd")
seeds=(0 1 2 3 4) # 5 seeds for each model-dataset pair
augs=("interaug" "no_interaug")
gpu_id=1

mkdir -p log

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seed in "${seeds[@]}"; do
            for aug in "${augs[@]}"; do

                if [ "$aug" == "interaug" ]; then
                    aug_flag="--interaug"
                    aug_suffix="aug"
                else
                    aug_flag="--no_interaug"
                    aug_suffix="noaug"
                fi

                # Standard run
                echo "Running $model on $dataset without LOSO (seed=$seed, $aug_suffix) on GPU $gpu_id"
                nohup python train_pipeline.py --model $model --dataset $dataset --gpu_id $gpu_id --seed $seed $aug_flag \
                    > log/${model}_${dataset}_std_${aug_suffix}_seed${seed}.log 2>&1 &
                wait

                # LOSO run
                echo "Running $model on $dataset with LOSO (seed=$seed, $aug_suffix) on GPU $gpu_id"
                nohup python train_pipeline.py --model $model --dataset $dataset --loso --gpu_id $gpu_id --seed $seed $aug_flag \
                    > log/${model}_${dataset}_loso_${aug_suffix}_seed${seed}.log 2>&1 &
                # Wait for both jobs to finish before next one
                wait
            done
        done
    done
done

# Ensemble TCN runs
ensemble_datasets=("bcic2a" "bcic2b" "hgd")
for dataset in "${ensemble_datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Training TCN ensemble on $dataset (seed=$seed)"
        nohup python ensemble_pipeline.py --datasets "$dataset" --seeds "$seed" \
            --data-root results --results-root results --train-file train.npz --test-file test.npz \
            --use-signal --target-length 128 --epochs 30 --batch-size 128 --gpu-id $gpu_id \
            > log/ensemble_${dataset}_seed${seed}.log 2>&1 &
        wait
    done
done

echo "All experiments completed!"
