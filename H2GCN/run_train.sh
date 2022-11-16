#!/bin/bash

for DATASET in <dataset_name>
do
    for SPLIT_ID in $( seq 0 9 )
    do
        echo "Split: $SPLIT_ID"
        for WEIGHT_DECAY in 1e-4 1e-3 1e-2
        do
            for LR in 1e-3 3e-3 1e-2 3e-2 1e-1
            do
                python h2gcn/run_experiments.py \
                    --model_type <MODEL_TYPE> \
                    --dataset_name ${DATASET} \
                    --lamda 30 \
                    --weight_decay ${WEIGHT_DECAY} \
                    --lr ${LR} \
                    --log_interval 1000 \
                    --split_id ${SPLIT_ID} >> ${DATASET}.txt
            done
        done
    done
done