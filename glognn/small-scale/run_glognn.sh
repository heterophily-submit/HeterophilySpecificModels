#!/bin/bash

export OMP_NUM_THREADS=8

for DATASET in chameleon chameleon_filtered_directed
do
    echo "DATASET=${DATASET}"
    for LR in 1e-3 3e-3 1e-2 3e-2 1e-1
    do
        echo "LR=${LR}"
        echo "LR=${LR}"  >> results/${DATASET}.txt
        for WD in 0 1e-5 5e-5 1e-4 1e-3
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            for SPLIT_ID in $( seq 0 9 )
            do
                echo "Split=${SPLIT_ID}"
                echo "Split=${SPLIT_ID}"  >> results/${DATASET}.txt
                python main.py \
                    --model mlp_norm \
                    --epochs 2000 \
                    --hidden 64 \
                    --lr ${LR} \
                    --dropout 0.0 \
                    --early_stopping 200 \
                    --weight_decay ${WD} \
                    --alpha 0.0 \
                    --beta 1.0 \
                    --gamma 0.0 \
                    --delta 0.0 \
                    --norm_layers 2 \
                    --orders 1 \
                    --orders_func_id 2 \
                    --norm_func_id 1 \
                    --dataset ${DATASET} \
                    --split ${SPLIT_ID}  >> results/${DATASET}.txt
            done
        done
    done
done

for DATASET in squirrel squirrel_filtered_directed roman_empire minesweeper questions amazon_ratings tolokers
do
    echo "DATASET=${DATASET}"
    for LR in 3e-4 1e-3 3e-3 1e-2 3e-2
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for WD in 0 1e-5 5e-5 1e-4 1e-3
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            for SPLIT_ID in $( seq 0 9 )
            do
                echo "Split=${SPLIT_ID}"
                echo "Split=${SPLIT_ID}" >> results/${DATASET}.txt
                python main.py \
                    --model mlp_norm \
                    --epochs 2000 \
                    --hidden 64 \
                    --lr ${LR} \
                    --dropout 0.8 \
                    --early_stopping 200 \
                    --weight_decay ${WD} \
                    --alpha 1.0 \
                    --beta 1.0 \
                    --gamma 0.0 \
                    --delta 0.0 \
                    --norm_layers 2 \
                    --orders 1 \
                    --orders_func_id 2 \
                    --norm_func_id 1 \
                    --dataset ${DATASET} \
                    --split ${SPLIT_ID} >> results/${DATASET}.txt
            done
        done
    done
done

for DATASET in roman_empire minesweeper questions amazon_ratings tolokers 
do
    echo "DATASET=${DATASET}"
    for LR in 3e-4 1e-3 3e-3 1e-2 3e-2
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for WD in 0 1e-5 5e-5 1e-4 1e-3
        do
            echo "WD=${WD}"
            echo "WD=${WD}" >> results/${DATASET}.txt
            for SPLIT_ID in $( seq 0 9 )
            do
                echo "Split=${SPLIT_ID}"
                echo "Split=${SPLIT_ID}" >> results/${DATASET}.txt
                python main.py \
                    --model mlp_norm \
                    --epochs 2000 \
                    --hidden 64 \
                    --lr ${LR} \
                    --dropout 0.8 \
                    --early_stopping 200 \
                    --weight_decay ${WD} \
                    --alpha 1.0 \
                    --beta 0.0 \
                    --gamma 0.0 \
                    --delta 1.0 \
                    --norm_layers 2 \
                    --orders 1 \
                    --orders_func_id 2 \
                    --norm_func_id 1 \
                    --dataset ${DATASET} \
                    --split ${SPLIT_ID} >> results/${DATASET}.txt
            done
        done
    done
done