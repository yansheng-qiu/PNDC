#!/bin/bash

dataname='BRATS2023'

datapath=./datasets/BRATS2023_Training_none_npy/


export CUDA_VISIBLE_DEVICES='1'


resume=output_tmi2023
savepath=test_2023

python train.py --batch_size=2 --datapath $datapath --savepath $savepath  --dataname $dataname  --resume $resume
