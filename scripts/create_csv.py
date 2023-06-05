# -*- coding: utf-8 -*-
"""
Created on Thu Mar 2 16:23:09 2023

@patch:
    2023.02.26
@author: Paul
@file: create_csv.py
@dependencies:
    env pt3.7 (my PC)
    python 3.7.13
    pytorch==1.7.1     py3.7_cuda110_cudnn8_0 pytorch
    torchaudio==0.7.2  py37 pytorch
    torchvision==0.8.2 py37_cu110 pytorch

Generate .csv or .txt files for training and testing use 
"""

import os
import csv
import random
import time
from datetime import date

DATASET = f"D:/Datasets/RADA/RD_JPG/"


def create_csv(num_train, total):
    train_file_name = DATASET + "csv_files/original_csv/" + f"train.csv"
    if os.path.isfile(f"{train_file_name}") is False:
        print(f"Creating '{train_file_name}' with {num_train} samples")
        with open(train_file_name, "w") as train_file:
            for i in range(1, num_train + 1):
                print(f"{i}.jpg,{i}.txt", file=train_file)

    test_file_name = DATASET + "csv_files/original_csv/" + f"test.csv"
    if os.path.isfile(f"{test_file_name}") is False:
        print(f"Creating '{test_file_name}' with {total - num_train} samples")
        with open(test_file_name, "w") as test_file:
            for i in range(num_train + 1, total + 1):
                print(f"{i}.jpg,{i}.txt", file=test_file)

    if os.path.isfile(f"{train_file_name}") is True and os.path.isfile(f"{test_file_name}") is True:
        print(f"Both '{train_file_name}' and '{test_file_name}' are already exist!\n")


indices = [i for i in range(1, 7193 + 1)]
def random_csv(num_train, num_test):
    # print(len(indices)) # 7193
    random.shuffle(indices)

    rand_train_file_name = DATASET + "csv_files/rand_csv/" + f"train.csv"
    rand_test_file_name  = DATASET + "csv_files/rand_csv/" + f"test.csv"

    with open(rand_train_file_name, "w") as train_file:
        for i in indices[0:num_train]:
            print(f"{i}.jpg,{i}.txt", file=train_file)
    
    with open(rand_test_file_name, "w") as test_file:
        for i in indices[-num_test::1]:
            print(f"{i}.jpg,{i}.txt", file=test_file)


def equal_splits_csv():
    print(f"")

    total_split = 6
    split = 0

    train_file_name = DATASET + "csv_files/equal_split_csv/" + f"train.csv"
    test_file_name  = DATASET + "csv_files/equal_split_csv/" + f"test.csv"
    
    for i in range(1, 7200 + 1):
        if i % total_split == split:
            print(f"{i}.jpg,{i}.txt")



if __name__ == "__main__":
    tic = time.perf_counter()

    num_train, total = 6000, 7193
    # create_csv(num_train=num_train, total=total)
    # random_csv(num_train=num_train, num_test=(total-num_train))

    equal_splits_csv()

    toc = time.perf_counter()
    duration = toc - tic
    print(f"duration: {duration:0.4f} seconds")

