#!/bin/bash

export OMP_NUM_THREADS=8

for DATASET in roman_empire
do
    for alpha in 0.0 0.5 1.0
    do
        for delta in 0.0 0.5 1.0
        do
            for beta in 0.0 0.5 1.0
            do
                echo "alpha=${alpha} beta=${beta} delta=${delta}"
                echo "alpha=${alpha} beta=${beta} delta=${delta}"  >> results/test_${DATASET}.txt
                python main.py \
                    --model mlp_norm \
                    --epochs 1000 \
                    --hidden 64 \
                    --lr 0.01 \
                    --dropout 0.0 \
                    --early_stopping 100 \
                    --weight_decay 5e-5 \
                    --alpha ${alpha} \
                    --beta ${beta} \
                    --gamma 0.0 \
                    --delta ${delta} \
                    --norm_layers 2 \
                    --orders 1 \
                    --orders_func_id 2 \
                    --norm_func_id 1 \
                    --dataset ${DATASET} \
                    --split 0  >> results/test_${DATASET}.txt
            done
        done
    done
done