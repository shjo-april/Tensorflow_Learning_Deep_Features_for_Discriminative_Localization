
import os
import glob

import numpy as np

def heatmap_to_bbox(heatmap, size, threshold = 127.5):
    xmin, ymin = size
    xmax, ymax = 0, 0

    h, w = heatmap.shape
    for y in range(h):
        for x in range(w):
            if heatmap[y, x] >= threshold:
                xmin = min(x, xmin)
                ymin = min(y, ymin)
                xmax = max(x, xmax)
                ymax = max(y, ymax)

    return [xmin, ymin, xmax, ymax]

def normalize(vector):
    min_value = np.min(vector)
    max_value = np.max(vector)
    
    vector -= min_value
    vector /= (max_value - min_value)

    return vector

def one_hot(label, classes):
    vector = np.zeros((classes), dtype = np.float32)
    vector[label] = 1
    return vector

def read_txt(txt_path, replace_dir):
    f = open(txt_path)
    lines = f.readlines()
    f.close()
    
    data_list = []
    for line in lines:
        image_path, label = line.strip().split('*')
        data_list.append([image_path.replace('../../../Dataset/flowers', replace_dir), int(label)])

    return data_list
