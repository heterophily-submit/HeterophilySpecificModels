#!/bin/bash

for DATASET in <dataset_name>
do
    for LR in 1e-3 3e-3 1e-2 3e-2 1e-1
    do
        echo "LR: ${LR}"
        echo "LR: ${LR}" >> ${DATASET}.txt
        for WEIGHT_DECAY in 1e-4 1e-3 1e-2
        do
            echo "LR: ${WEIGHT_DECAY}"
            echo "LR: ${WEIGHT_DECAY}" >> ${DATASET}.txt
            for SPLIT_ID in $( seq 0 9 )
                do
                echo "Split: $SPLIT_ID"
                echo "Split: $SPLIT_ID"  >> ${DATASET}.txt
                python node_classification.py \
                    --model_type GraphSage \
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