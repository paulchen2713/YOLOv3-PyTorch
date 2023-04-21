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

import csv
import random
import time
from datetime import date

DATASET = f"D:/Datasets/RADA/RD_JPG/"


def create_csv(split, last):
    with open(DATASET + f"train.csv", "w") as train_file:
        for i in range(1, split + 1):
            print(f"{i}.jpg,{i}.txt", file=train_file)

    with open(DATASET + f"test.csv", "w") as test_file:
        for i in range(split + 1, last + 1):
            print(f"{i}.jpg,{i}.txt", file=test_file)


indices = [i for i in range(1, 7193 + 1)]
def random_csv(num_train, num_test):
    print(len(indices)) # 7193
    random.shuffle(indices)

    with open(DATASET + "csv_files/rand_csv/" + f"train.csv", "w") as train_file:
        for i in indices[0:num_train]:
            print(f"{i}.jpg,{i}.txt", file=train_file)

    with open(DATASET + "csv_files/rand_csv/" + f"test.csv", "w") as test_file:
        for i in indices[-num_test::1]:
            print(f"{i}.jpg,{i}.txt", file=test_file)



if __name__ == "__main__":
    tic = time.perf_counter()

    split, last = 6000, 7193
    # create_csv(split=split, last=last)
    # print(f"Creating tran.csv with {split} samples and test.csv with {last - split} samples")
    random_csv(num_train=split, num_test=last-split)

    toc = time.perf_counter()
    duration = toc - tic
    print(f"duration: {duration:0.4f} seconds")

