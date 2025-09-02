#!/bin/bash

dataname='BRATS2023'

datapath=./BRATS2023_Training_none_npy/

export CUDA_VISIBLE_DEVICES='1'

savepath=output_2023
python train_PNDC.py --batch_size=2 --datapath $datapath --savepath $savepath --num_epochs 600 --lr 2e-4 --region_fusion_start_epoch 0 --dataname $dataname 

