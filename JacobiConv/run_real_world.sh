#!/bin/bash

for DATASET in <dataset_name>
do
    for SPLIT_ID in $( seq 0 9 )
    do
        python RealWorld.py \
            --repeat 1 \
            --dataset $DATASET \
            --split default \
            --optruns 20 \
            --name ${DATASET}_${SPLIT_ID}_opt \
            --split_id ${SPLIT_ID} >> ${DATASET}.txt
    done
done