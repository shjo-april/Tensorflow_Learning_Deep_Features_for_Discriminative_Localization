import numpy as np
import tensorflow as tf

import vgg_16.VGG16 as vgg

from Define import *

def Global_Average_Pooling(x):    
    pool_size = np.shape(x)[1:3][::-1]
    return tf.layers.average_pooling2d(inputs = x, pool_size = pool_size, strides = 1)

def Visualize(single_conv, fc_w, fc_b = None):
    h, w, c = single_conv.shape

    heatmap_conv = tf.reshape(single_conv, [h * w, c])
    heatmap_fc_w = tf.reshape(fc_w, [c, CLASSES])
    heatmap_flat = tf.matmul(heatmap_conv, heatmap_fc_w)

    if fc_b != None:
        heatmap_flat += fc_b

    heatmap_op = tf.reshape(heatmap_flat, [h, w, CLASSES])
    return heatmap_op

def fine_tuning(input_var, is_training):
    x = input_var - VGG_MEAN

    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        x = vgg.vgg_16(x, num_classes=1000, is_training=is_training, dropout_keep_prob=0.5)
    
    feature_maps = x
    x = Global_Average_Pooling(x)
    x = tf.contrib.layers.flatten(x)

    logits = tf.layers.dense(x, CLASSES, use_bias = False, name = 'logits')
    predictions = tf.nn.softmax(logits)

    return logits, predictions, feature_maps
