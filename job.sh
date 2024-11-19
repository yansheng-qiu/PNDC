#!/bin/bash

dataname='BRATS2023'

datapath=./datasets/BRATS2023_Training_none_npy/


export CUDA_VISIBLE_DEVICES='1'


resume=model_last.pth
savepath=test_2023

python train.py --batch_size=2 --datapath $datapath --savepath $savepath  --dataname $dataname  --resume $resume
