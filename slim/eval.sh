#!/bin/sh
CHECKPOINT_FILE=/home/zh/Downloads/train/ # Example
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=/home/zh/Downloads/xiaoying \
    --dataset_name=xiaoying \
    --dataset_split_name=validation \
    --model_name=mobilenet