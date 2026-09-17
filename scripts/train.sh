#bin/bash

 CUDA_VISIBLE_DEVICES=1 nohup  python3 train.py \
 --dataset Synapse \
 --vit_name R50-ViT-B_16 \
 --max_epochs 200 \
 
