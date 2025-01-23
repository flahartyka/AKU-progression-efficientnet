#!/bin/bash

# Set CUDA device
#export CUDA_VISIBLE_DEVICES="1"

source myconda
mamba activate base

# Define variables
arch="vit_h"  # Change this value as needed
finetune_type="adapter"
dataset_name="cervical-dataset"  # Set the name here
targets='combine_all' # make it as binary segmentation 'multi_all' for multi cls segmentation
# Construct train and validation image list paths
img_folder="/"  # Assuming this is the folder where images are stored
train_img_list="datasets/train_5shot.csv"
val_img_list="datasets/val_5shot.csv"


# Construct the checkpoint directory argument
dir_checkpoint="2D-SAM_${arch}_decoder_${finetune_type}_${dataset_name}_noprompt"
CUDA_LAUNCH_BLOCKING=1

#cd SAM-datasets

# Run the Python script
python SingleGPU_train_finetune_noprompt.py \
    -if_warmup True \
    -finetune_type "$finetune_type" \
    -arch "$arch" \
    -targets "$targets" \
    -if_mask_decoder_adapter True \
    -img_folder "$img_folder" \
    -mask_folder "$img_folder" \
    -sam_ckpt "sam_vit_h_4b8939.pth" \
    -dataset_name "$dataset_name" \
    -dir_checkpoint "$dir_checkpoint" \
    -train_img_list "$train_img_list" \
    -val_img_list "$val_img_list"

