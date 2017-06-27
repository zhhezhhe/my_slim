#!/bin/sh
CHECKPOINT_DIR=checkpoints
mkdir ${CHECKPOINT_DIR}
wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
tar -xvf inception_v4_2016_09_09.tar.gz
mv inception_v4.ckpt ${CHECKPOINT_DIR}
rm inception_v4_2016_09_09.tar.gz
