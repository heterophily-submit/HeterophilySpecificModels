#!/bin/bash

export OMP_NUM_THREADS=8 

for DATASET in <dataset_name>
do
    for LR in 1e-3 3e-3 1e-2 3e-2 1e-1
    do
        echo "Lr ${LR}"
        echo "Lr ${LR}" >> ${DATASET}.txt
        for WD in 1e-4 1e-3 1e-2
        do
            echo "Weight decay ${WD}"
            echo "Weight decay ${WD}" >> ${DATASET}.txt
            for i in {0..9} 
            do
                echo "Split ${i}"
                echo "Split ${i}" >> ${DATASET}.txt
                python train.py \
                    --dataset ${DATASET} \
                    --dropout 0.0 \
                    --eps 0.3 \
                    --lr ${LR} \
                    --hidden 128 \
                    --patience 100 \
                    --epochs 1000 \
                    --weight_decay ${WD} \
                    --splits_file_path ${i} >> ${DATASET}.txt
            done
        done
    done
done