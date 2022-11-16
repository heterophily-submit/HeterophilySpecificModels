#!/bin/bash

export OMP_NUM_THREADS=8

for DATASET in workers
do
    for LR in 1e-3 3e-3 1e-2 3e-2 1e-1
    do
        echo "LR: ${LR}"
        echo "LR: ${LR}" >> ${DATASET}.txt
        for WD in 1e-4 1e-3 1e-2
        do
            echo "weight decay ${WD}"
            echo "weight decay ${WD}" >> exp_results_${DATASET}.txt
            for SPLIT_ID in {0..9}
            do
                echo "Split: $SPLIT_ID"
                echo "Split: $SPLIT_ID"  >> ${DATASET}.txt
                python3 main.py --model mlp_norm --epochs 1000 --hidden 64 --lr ${LR} --dropout 0.0 --early_stopping 100 --weight_decay ${WD} --alpha 0.0 --beta 1.0 --gamma 0.0 --delta 1.0 --norm_layers 2 --orders 2 --orders_func_id 2 --norm_func_id 1 --dataset ${DATASET} --split ${SPLIT_ID} >> exp_results_${DATASET}.txt
            done
        done
    done
done
