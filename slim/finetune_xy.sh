DATASET_DIR=/home/zh/Downloads/xiaoying
TRAIN_DIR=/home/zh/Downloads/train
CHECKPOINT_PATH=/home/zh/Downloads/MobileNet/pretrained/model.ckpt-906808
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=xiaoying \
    --dataset_split_name=train \
    --model_name=mobilenet \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=MobileNet/fc_16 \
    --trainable_scopes=MobileNet/fc_16