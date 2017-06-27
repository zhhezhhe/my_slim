# -*- coding: UTF-8 -*-
#encoding=utf-8

import sys
sys.path.append("/media/zh/E/models/slim")

import os
import numpy as np
import tensorflow as tf

from nets import inception
from datasets import imagenet
from preprocessing import inception_preprocessing

slim = tf.contrib.slim

with tf.Session() as sess:

    arg_scope = inception.inception_v4_arg_scope()
    input_tensor = tf.placeholder(tf.float32, [None, 299, 299, 3])

    with slim.arg_scope(arg_scope):
        logits, _ = inception.inception_v4(inputs=input_tensor,is_training=False)

    saver = tf.train.Saver()
    saver.restore(sess,'checkpoints/inception_v4.ckpt')

    graph = tf.get_default_graph()
    tensor1 = graph.get_tensor_by_name("InceptionV4/Logits/AvgPool_1a/AvgPool:0")

    print tensor1

    # for node in graph.as_graph_def().node:
    #   print(node.name)