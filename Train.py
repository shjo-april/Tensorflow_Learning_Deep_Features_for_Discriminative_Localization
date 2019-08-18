import cv2
import time
import random

import numpy as np
import tensorflow as tf

from Utils import *
from Define import *
from fine_tuning import *

# 1. load dataset
train_data = read_txt('./dataset/train.txt', REPLACE_DIR)
valid_data = read_txt('./dataset/valid.txt', REPLACE_DIR)
test_data  = read_txt('./dataset/test.txt', REPLACE_DIR)

print('[i] Dataset')
print('[i] Train : {}'.format(len(train_data)))
print('[i] Valid : {}'.format(len(valid_data)))
print('[i] Test  : {}'.format(len(test_data)))
print()

# 2. build model
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
label_var = tf.placeholder(tf.float32, [None, CLASSES])
is_training = tf.placeholder(tf.bool)

logits, prediction_op, feature_maps = fine_tuning(input_var, is_training)

# 3. loss & accuracy
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = label_var))

correct_op = tf.equal(tf.argmax(predictions, axis = 1), tf.argmax(label_var, axis = 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

# 4. optimizer
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss_op)

# 5. train
sess = tf.Session()
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

vgg_vars = []
for var in tf.trainable_variables():
    if 'vgg_16' in var.name:
        vgg_vars.append(var)

pretrain_saver = tf.train.Saver(var_list = vgg_vars)
pretrain_saver.restore(sess, './vgg_16_model/vgg_16.ckpt')

train_iteration = len(train_data) // BATCH_SIZE
valid_iteration = len(valid_data) // BATCH_SIZE

max_iteration = train_iteration * MAX_EPOCH

best_valid_accuracy = 0.0

train_loss_list = []
train_accuracy_list = []

for iter in range(1, max_iteration + 1):
    
    batch_data = random.sample(train_data, BATCH_SIZE)

    batch_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)
    batch_label_data = np.zeros((BATCH_SIZE, CLASSES), dtype = np.float32)

    for index, data in enumerate(batch_data):
        image_path, label = data

        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        batch_image_data[index] = image.astype(np.float32)
        batch_label_data[index] = one_hot(label, CLASSES)
        
    _, loss, accuracy = sess.run([train_op, loss_op, accuracy_op], feed_dict = {input_var : batch_image_data, label_var : batch_label_data, is_training : True})

    train_loss_list.append(loss)
    train_accuracy_list.append(accuracy)
    
    if iter % LOG_ITERATION == 0:
        train_loss = np.mean(train_loss_list)
        train_accuracy = np.mean(train_accuracy_list)

        print('[i] iter : {}, loss = {:.5f}, accuracy = {:.2f}'.format(iter, train_loss, train_accuracy))

        train_loss_list = []
        train_accuracy_list = []

    if iter % VALID_ITERATION == 0:
        valid_accuracy_list = []

        for iter in range(valid_iteration):
            batch_data = valid_data[iter * BATCH_SIZE : (iter + 1) * BATCH_SIZE]

            batch_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.float32)
            batch_label_data = np.zeros((BATCH_SIZE, CLASSES), dtype = np.float32)

            for index, data in enumerate(batch_data):
                image_path, label = data

                image = cv2.imread(image_path)
                image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

                batch_image_data[index] = image.astype(np.float32)
                batch_label_data[index] = one_hot(label, CLASSES)
                
            accuracy = sess.run(accuracy_op, feed_dict = {input_var : batch_image_data, label_var : batch_label_data, is_training : False})
            valid_accuracy_list.append(accuracy)

        valid_accuracy = np.mean(valid_accuracy_list)

        if best_valid_accuracy < valid_accuracy:
            best_valid_accuracy = valid_accuracy
            saver.save(sess, './model/VGG16_{}.ckpt'.format(iter))

        print('[i] valid accuracy : {:.2f}, best valid accuracy : {:.2f}'.format(valid_accuracy, best_valid_accuracy))
