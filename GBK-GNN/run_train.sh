#!/bin/bash

export OMP_NUM_THREADS=8

for DATASET in chameleon_directed chameleon_filtered_directed squirrel_directed squirrel_filtered_directed roman_empire minesweeper questions amazon_ratings tolokers
do
    echo "DATASET=${DATASET}"
    for LR in 0.001 0.003 0.01 0.03 0.1
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for WD in 0.0 1e-5 1e-4 1e-3
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            for SPLIT_ID in $( seq 0 9 )
            do
                echo "Split=${SPLIT_ID}"
                echo "Split=${SPLIT_ID}" >> results/${DATASET}.txt
                python node_classification.py \
                    --model_type GraphSage \
                    --dataset_name ${DATASET} \
                    --lamda 30 \
                    --weight_decay ${WD} \
                    --lr ${LR} \
                    --log_interval 1000 \
                    --split_id ${SPLIT_ID} >> results/${DATASET}.txt
            done
        done
    done
done
