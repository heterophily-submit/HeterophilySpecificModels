#!/bin/bash

mkdir -p results

for DATASET in chameleon_directed chameleon_filtered_directed squirrel squirrel_filtered_directed roman_empire minesweeper questions amazon_ratings tolokers
do
    echo "DATASET=${DATASET}"
    for LR in 0.001 0.003 0.01 0.03
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for WD in 0 1e-6 1e-5 1e-4
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            for SPLIT_ID in $( seq 0 9 )
            do
                echo "Split=${SPLIT_ID}"
                echo "Split=${SPLIT_ID}" >> results/${DATASET}.txt
                python run_experiments.py CPGNN planetoid \
                --network_setup M64-D-R-MO-E-BP2 \
                --dropout 0.0 \
                --dataset $DATASET \
                --dataset_path data \
                --epochs 1000 \
                --lr ${LR} \
                --l2_regularize_weight ${WD} \
                --early_stopping 100 \
                --split_id $SPLIT_ID >> results/${DATASET}.txt
            done
        done
    done
done
