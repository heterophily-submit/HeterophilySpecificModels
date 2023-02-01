#!/bin/bash

# set number in [1, 8]
export OMP_NUM_THREADS=8

for DATASET in chameleon_directed chameleon_filtered_directed
do
    echo "DATASET=${DATASET}"
    for LR in 0.003 0.01 0.03 0.1
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for WD in 0 1e-5 1e-4 1e-3 
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            for SPLIT_ID in $( seq 0 9 ) 
            do
                echo "Split=${SPLIT_ID}"
                echo "Split=${SPLIT_ID}" >> results/${DATASET}.txt
                python train.py \
                    --dataset ${DATASET} \
                    --dropout 0.5 \
                    --eps 0.4 \
                    --lr ${LR} \
                    --hidden 32 \
                    --patience 100 \
                    --epochs 500 \
                    --weight_decay ${WD} \
                    --splits_file_path ${SPLIT_ID} >> results/${DATASET}.txt
            done
        done
    done
done

for DATASET in squirrel_directed squirrel_filtered_directed  roman_empire minesweeper questions amazon_ratings workers
do
    echo "DATASET=${DATASET}"
    for LR in 0.003 0.01 0.03 0.1
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for WD in 0 1e-5 1e-4 1e-3 
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            for SPLIT_ID in $( seq 0 9 ) 
            do
                echo "Split=${SPLIT_ID}"
                echo "Split=${SPLIT_ID}" >> results/${DATASET}.txt
                python train.py \
                    --dataset ${DATASET} \
                    --dropout 0.5 \
                    --eps 0.3 \
                    --lr ${LR} \
                    --hidden 32 \
                    --patience 100 \
                    --epochs 500 \
                    --weight_decay ${WD} \
                    --splits_file_path ${SPLIT_ID} >> results/${DATASET}.txt
            done
        done
    done
done