#!/bin/bash

for DATASET in <dataset_name>
do
    for LR in 1e-3 3e-3 1e-2 3e-2 1e-1
    do
        for WD in 1e-4 1e-3 1e-2
        do
            for NUM_LAYERS in 3
            do
                python train_fsgnn.py \
                    --seed 42 \
                    --steps 1500 \
                    --log-freq 100 \
                    --num-layers ${NUM_LAYERS} \
                    --hidden-dim 64 \
                    --patience 100 \
                    --dataset ${DATASET} \
                    --layer-norm 1 \
                    --lr ${LR} \
                    --weight-decay ${WD} \
                    --dropout 0.0 \
                    --transform \
                    --feat-type all
            done
        done
    done
done