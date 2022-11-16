#!/bin/bash

RPMAX=10

for DATASET in <dataset_name>
do
        for WD in 1e-4 1e-3 1e-2
        do
                echo "WD: ${WD}"
                echo "WD: ${WD}" >> ${DATASET}.txt
                for LR in 1e-3 3e-3 1e-2 3e-2 1e-1
                do
                        echo "LR: ${LR}"
                        echo "LR: ${LR}" >> ${DATASET}.txt
                        python train_model.py --RPMAX $RPMAX \
                                --net GPRGNN \
                                --train_rate 0.6 \
                                --val_rate 0.2 \
                                --dataset ${DATASET} \
                                --lr ${LR} \
                                --alpha 0.9 \
                                --weight_decay ${WD} >> ${DATASET}.txt 
                done
        done
done
