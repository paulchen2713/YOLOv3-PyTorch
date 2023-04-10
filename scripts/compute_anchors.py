# -*- coding: utf-8 -*-
"""
Created on Sat Apr 08 20:49:26 2023

@patch: 2023.04.08
@author: Paul
@file: compute_anchors.py
@dependencies:
    env pt3.7 (my PC)
    python 3.7.13
    pytorch==1.7.1     py3.7_cuda110_cudnn8_0 pytorch
    torchaudio==0.7.2  py37 pytorch
    torchvision==0.8.2 py37_cu110 pytorch
    pillow==8.1.0

Recompute YOLO anchors
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

import json
import os
from os import listdir
import time
from datetime import date


# directory names, number of directorie: 30
dir_names = ['2019-09-16-12-52-12', '2019-09-16-12-55-51', '2019-09-16-12-58-42', '2019-09-16-13-03-38', '2019-09-16-13-06-41', 
             '2019-09-16-13-11-12', '2019-09-16-13-13-01', '2019-09-16-13-14-29', '2019-09-16-13-18-33', '2019-09-16-13-20-20', 
             '2019-09-16-13-23-22', '2019-09-16-13-25-35', '2020-02-28-12-12-16', '2020-02-28-12-13-54', '2020-02-28-12-16-05', 
             '2020-02-28-12-17-57', '2020-02-28-12-20-22', '2020-02-28-12-22-05', '2020-02-28-12-23-30', '2020-02-28-13-05-44', 
             '2020-02-28-13-06-53', '2020-02-28-13-07-38', '2020-02-28-13-08-51', '2020-02-28-13-09-58', '2020-02-28-13-10-51', 
             '2020-02-28-13-11-45', '2020-02-28-13-12-42', '2020-02-28-13-13-43', '2020-02-28-13-14-35', '2020-02-28-13-15-36']


# number of images / labels in each directory, total number of labels: 7193
num_of_images = [286, 273, 304, 327, 218, 219, 150, 208, 152, 174, 
                 174, 235, 442, 493, 656, 523, 350, 340, 304, 108, 
                 129, 137, 171, 143, 104, 81, 149, 124, 121, 98]


def read_width_height(file_name):
    count = 0
    for dir_name in dir_names: # [23:24]: # 
        # print(f"current directory: {dir_name}")

        # set the file path
        file_index = ["range_doppler_light.json", "range_angle_light.json"]
        with open(f"D:/Datasets/CARRADA/{dir_name}/annotations/box/" + f"{file_index[0]}", "r") as json_file:
            # read out all the bbox labels 
            data = json.loads(json_file.read())
        
        # extract all keys from the dict, and store them in a list()
        all_keys = list(data.keys())

        for key in all_keys: # [62:63]: # 
            # print(f"frame name: \"{key}\"")
            count += 1
            print(count)
            
            # in each rd_matrix / image it may contain 1~3 possible targets
            for index in range(0, len(data[key]['boxes'])):
                # class_index = data[key]['labels'][index] - 1
                x_min, y_min, x_max, y_max = data[key]['boxes'][index][0:4]

                h = x_max - x_min
                w = y_max - y_min
                y = int((y_min + y_max) / 2)
                x = int((x_min + x_max) / 2)

                # rescale to (0, 1)
                x, h = x / 256, h / 256
                y, w = y / 64, w / 64
                # x_min, y_min, x_max, y_max = x_min / 256, y_min / 64, x_max / 256, y_max / 64
                with open(file_name, "a") as label_txt_file:
                    print(f"{w} {h}", file=label_txt_file)


def load_data(file_name):
    data = []
    with open(file_name, "r", encoding="utf-8") as text_file:
        # print(f"current file: {file_path}")
        lines = text_file.readlines()
        for line in lines:
            myarray = np.fromstring(line, dtype=float, sep=' ')
            data.append(myarray)
            # print(line)
            # print(myarray) 
    return data


class K_Means:
    def __init__(self, k=9, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


if __name__ == '__main__':
    tic = time.perf_counter()

    file_name = f"D:/Datasets/RADA/RD_JPG/width_heights.txt"
    # read_width_height(file_name=file_name)
    width_heights = load_data(file_name=file_name)
    # width_heights = np.array(width_heights)
    # print(len(width_heights))
    print(type(width_heights))
    # print(width_heights[0][1])


   

    toc = time.perf_counter()
    duration = toc - tic
    print(f"duration: {duration:0.4f} seconds")









