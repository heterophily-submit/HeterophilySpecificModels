#!/bin/bash

export OMP_NUM_THREADS=8

for DATASET in chameleon_directed chameleon_filtered_directed
do
    echo "DATASET=${DATASET}"
    for LR in 0.001 0.003 0.01 0.02 0.04
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for WD in 0.0 1e-5 3e-5 1e-4 3e-4
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            python train_fsgnn.py \
                --seed 42 \
                --steps 1500 \
                --log-freq 100 \
                --num-layers 3 \
                --hidden-dim 64 \
                --patience 100 \
                --dataset ${DATASET} \
                --layer-norm 1 \
                --lr ${LR} \
                --weight-decay ${WD} \
                --dropout 0.5 \
                --feat-type all >> results/${DATASET}.txt
        done
    done
done

for DATASET in squirrel_directed squirrel_filtered_directed roman_empire minesweeper questions amazon_ratings workers
do
    echo "DATASET=${DATASET}"
    for LR in 0.001 0.003 0.01 0.02 0.04
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for WD in 0.0 1e-5 3e-5 1e-4 3e-4
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            python train_fsgnn.py \
                --seed 42 \
                --steps 1500 \
                --log-freq 100 \
                --num-layers 3 \
                --hidden-dim 64 \
                --patience 100 \
                --dataset ${DATASET} \
                --layer-norm 1 \
                --lr ${LR} \
                --weight-decay ${WD} \
                --dropout 0.7 \
                --feat-type all >> results/${DATASET}.txt
        done
    done
done