# -*- coding: utf-8 -*-
"""
Created on Sun Aug 06 13:50:42 2023

@patch: 2023.08.06
@author: Paul
@file: read_points.py
@dependencies:
    envs        pt3.7
    python 3.7.13
    pytorch==1.7.1     py3.7_cuda110_cudnn8_0 pytorch
    torchaudio==0.7.2  py37 pytorch
    torchvision==0.8.2 py37_cu110 pytorch
    matplotlib==3.3.4
    scipy==1.7.3
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import json


# set the dataset path
DATASET = 'D:/Datasets/CARRADA/'

# directory names, number of directorie: 30
dir_names = ['2019-09-16-12-52-12', '2019-09-16-12-55-51', '2019-09-16-12-58-42', '2019-09-16-13-03-38', '2019-09-16-13-06-41', 
             '2019-09-16-13-11-12', '2019-09-16-13-13-01', '2019-09-16-13-14-29', '2019-09-16-13-18-33', '2019-09-16-13-20-20', 
             '2019-09-16-13-23-22', '2019-09-16-13-25-35', '2020-02-28-12-12-16', '2020-02-28-12-13-54', '2020-02-28-12-16-05', 
             '2020-02-28-12-17-57', '2020-02-28-12-20-22', '2020-02-28-12-22-05', '2020-02-28-12-23-30', '2020-02-28-13-05-44', 
             '2020-02-28-13-06-53', '2020-02-28-13-07-38', '2020-02-28-13-08-51', '2020-02-28-13-09-58', '2020-02-28-13-10-51', 
             '2020-02-28-13-11-45', '2020-02-28-13-12-42', '2020-02-28-13-13-43', '2020-02-28-13-14-35', '2020-02-28-13-15-36']


annotation_type = ['sparse', 'dense']
type_name = annotation_type[1]

seq_path = DATASET + dir_names[23] + f'/annotations/{type_name}/'
print(seq_path)

sub_dir = f'000120'

radar_signal_type = ['range_doppler', 'range_angle']
points_path = seq_path + sub_dir + f'/{radar_signal_type[0]}.npy'

points_matrix = np.load(points_path)

store_folder = ['images', 'mats', 'sparse', 'dense']
store_path = f"D:/Datasets/RADA/RD_JPG/{type_name}/"


print(f"{type_name}_points_matrix.shape: {points_matrix.shape}")
for index, point in enumerate(points_matrix):
    print(f"{index}-th point shape: {point.shape}")
    print(point)
    print("-"*100)
    plt.matshow(point, interpolation="nearest")
    plt.axis('off')
    # plt.show()
    plt.savefig(store_path + 'images/' + f'{type_name}_{sub_dir}_{index}.jpg', bbox_inches='tight', pad_inches=0)
    plt.clf()  # clears the entire current figure 
    plt.close(plt.gcf())

    scipy.io.savemat(store_path + 'mats/' + f'{type_name}_{sub_dir}_{index}.mat', {f'{type_name}_{sub_dir}_{index}': point})


