#!/bin/bash

export OMP_NUM_THREADS=8

RPMAX=10

for DATASET in chameleon_directed chameleon_filtered_directed
do
    echo "DATASET=${DATASET}"
    for LR in 0.005 0.01 0.05 0.1
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for WD in 0 1e-5 1e-4 1e-3
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            python train_model.py --RPMAX $RPMAX \
                --net GPRGNN \
                --train_rate 0.6 \
                --val_rate 0.2 \
                --dataset ${DATASET} \
                --lr ${LR} \
                --alpha 1.0 \
                --dprate 0.7 \
                --weight_decay ${WD} >> results/${DATASET}.txt
        done
    done
done

for DATASET in squirrel_directed squirrel_filtered_directed roman_empire minesweeper questions amazon_ratings workers
do
    echo "DATASET=${DATASET}"
    for LR in 0.005 0.01 0.05 0.1
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for WD in 0 1e-5 1e-4 1e-3
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            python train_model.py --RPMAX $RPMAX \
                --net GPRGNN \
                --train_rate 0.6 \
                --val_rate 0.2 \
                --dataset ${DATASET} \
                --lr ${LR} \
                --alpha 0 \
                --dprate 0.7 \
                --weight_decay ${WD} >> results/${DATASET}.txt
            done
        done
    done
done
