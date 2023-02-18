#!/bin/bash

export OMP_NUM_THREADS=8

for DATASET in chameleon_directed chameleon_filtered_directed
do
    echo "DATASET=${DATASET}"
    for LR in 0.01 0.03 0.05 0.1
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for WD in 0.0 1e-5 1e-4 5e-4
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            for SPLIT_ID in $( seq 0 9 )
            do
                echo "Split=${SPLIT_ID}"
                echo "Split=${SPLIT_ID}" >> results/${DATASET}.txt
                python RealWorld.py \
                    --dataset $DATASET \
                    --split default \
                    --lr ${LR} \
                    --wd ${WD} \
                    --dpb 0.6 \
                    --dpt 0.5 \
                    --a 0 \
                    --alpha 2.0 \
                    --b 0 \
                    --name ${DATASET}_${SPLIT_ID}_opt \
                    --split_id ${SPLIT_ID} >> results/${DATASET}.txt
            done
        done
    done
done

for DATASET in squirrel_directed squirrel_filtered_directed roman_empire minesweeper questions amazon_ratings tolokers
do
    echo "DATASET=${DATASET}"
    for LR in 0.01 0.03 0.05 0.1
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for WD in 0.0 1e-6 1e-5 1e-4
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            for SPLIT_ID in $( seq 0 9 )
            do
                echo "Split=${SPLIT_ID}"
                echo "Split=${SPLIT_ID}" >> results/${DATASET}.txt
                python RealWorld.py \
                    --dataset $DATASET \
                    --split default \
                    --lr ${LR} \
                    --wd ${WD} \
                    --dpb 0.4 \
                    --dpt 0.1 \
                    --a 0.5 \
                    --alpha 2.0 \
                    --b 0.25 \
                    --name ${DATASET}_${SPLIT_ID}_opt \
                    --split_id ${SPLIT_ID} >> results/${DATASET}.txt
            done
        done
    done
done
