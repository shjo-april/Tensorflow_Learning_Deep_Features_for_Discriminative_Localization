import os
import cv2
import numpy as np
import tensorflow as tf

from Utils import *
from Define import *
from fine_tuning import *

# 1. load dataset
test_data  = read_txt('./dataset/test.txt', REPLACE_DIR)
np.random.shuffle(test_data)

# 2. build model
input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

logits, prediction_op, feature_maps = fine_tuning(input_var, False)

# 3. heatmap
fc_kernel = None
fc_bias = None

for var in tf.trainable_variables():
    if 'logits' in var.name:
        if 'kernel' in var.name:
            fc_kernel = var
        else:
            fc_bias = var

heatmaps_op = Visualize(feature_maps[0], fc_kernel, fc_bias)

# 4. test
sess = tf.Session()
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
saver.restore(sess, './model/VGG16.ckpt')

for data in test_data[:20]:
    image_path, label = data

    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    predictions, heatmap = sess.run([prediction_op, heatmaps_op], feed_dict = {input_var : [image]})
    
    prediction = predictions[0]
    pred_index = np.argmax(prediction)

    class_name = CLASS_NAMES[pred_index]
    class_prob = prediction[pred_index] * 100

    # decode heatmaps
    heatmap = (normalize(heatmap[:, :, pred_index]) * 255.).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (IMAGE_WIDTH, IMAGE_HEIGHT))
    show_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # decode bbox
    pred_bbox = heatmap_to_bbox(heatmap, (IMAGE_WIDTH, IMAGE_HEIGHT), 255. * 0.4)

    # Localization
    xmin, ymin, xmax, ymax = pred_bbox
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Classification
    string = '{}_{:.2f}%'.format(class_name, class_prob)
    cv2.putText(image, string, (0, 10), 1, 1, (0, 255, 0), 2)

    # Display
    show_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH * 2, IMAGE_CHANNEL), dtype = np.uint8)
    show_image[:, :IMAGE_WIDTH, :] = image
    show_image[:, IMAGE_WIDTH:, :] = show_heatmap

    # cv2.imshow('show', show_image)
    # cv2.waitKey(0)

    cv2.imwrite('./results/' + os.path.basename(image_path), show_image)
