# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:09:11 2022

@author: Paul
@file: iterate_all_files.py
@dependencies:
    env pt3.7 (my PC)
    python 3.7.13
    pytorch==1.7.1     py3.7_cuda110_cudnn8_0 pytorch
    torchaudio==0.7.2  py37 pytorch
    torchvision==0.8.2 py37_cu110 pytorch
"""

# import the required libraries
import torchvision.transforms as T # for resizing the images
from PIL import Image              # for loading and saving the images
import os
from os import listdir
import shutil
import time

folder_name = ['RD_Pascal_VOC', 'RD_YOLO', 'RD_COCO', 'RD', 'RD2', 'RD3', 'RA']

# set the dataset path
DATASET = f'D:/Datasets/CARRADA2/{folder_name[6]}/'
DATASET2 = f'D:/Datasets/RADA/'

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

# e.g. read "validated_seqs.txt"
def read_txt_file(file_name=""):
    dir_names = list()
    with open(DATASET + file_name, "r") as seqs_file:
        dir_names = seqs_file.read().splitlines()
    return dir_names
# temp = read_txt_file("validated_seqs.txt")


def delete_useless_files():
    # count = 1
    for dir_name in dir_names: # [23:24]: # 
        # print(f"current directory: {dir_name}")

        # set the file path
        seq_path = DATASET + dir_name + '/labels/'
        # print(f"current seq path: {seq_path}")
        isFound = False
        for labels in os.listdir(seq_path):
            # check if the labels ends with .txt
            if (labels.endswith(".txt")):
                # print(f"label type: {type(labels)}") # <class 'str'>
                if labels[0:5] == "0000_":
                    isFound = True
                    print(seq_path + labels)
                    if isFound == True: os.remove(seq_path + labels)
                if labels[0:2] == "RD":
                    isFound = True
                    print(seq_path + labels)
                    if isFound == True: os.remove(seq_path + labels)
                if labels[0:4] == "log_":
                    isFound = True
                    print(seq_path + labels)
                    if isFound == True: os.remove(seq_path + labels)
                # print(count)
                # count += 1
    if isFound == False: 
        print("It's clear!")


def copy_and_rename_labels():
    count = 0
    for dir_name in dir_names: # [23:24]: # 
        # print(f"current directory: {dir_name}")

        # set the file path
        seq_path = DATASET + dir_name + '/labels/'
        # print(f"current seq path: {seq_path}")

        for labels in os.listdir(seq_path):
            # check if the labels ends with .txt
            if (labels.endswith(".txt")):
                if labels[0:5] == "0000_" or labels[0:2] == "RD": print("Error!")
                # shutil.copyfile(seq_path + labels, DATASET + 'labels/' + f'{count}.txt')
                count += 1
                print(count)
                
    if count != 7193: print("Error!")


def copy_and_rename_labels2():
    count = 0
    # set the file path
    label_path = 'C:/Users/Paul/Downloads/YOLOv3/src/'

    for labels in os.listdir(label_path):
        # check if the labels ends with .txt
        if (labels.endswith(".txt")):
            # print(f"{labels}")
            count += 1
            print(count)
            dest_path = 'C:/Users/Paul/Downloads/YOLOv3/dest/'
            shutil.copyfile(label_path + labels, dest_path + f'{count}.txt')
            if count == 7194: break
        


def copy_and_rename_images():
    count = 0
    # set the file path
    folder_name2 = ['RD_64', 'RD_256', 'RD_416', 'RD_YOLO']
    folder_index = 3
    # src_path = DATASET2 + 'RD_PNG/' + f'{folder_name2[folder_index]}/images/'
    src_path = f'D:/Datasets/RD_YOLO/images/'
    print(f"current source path: {src_path}")

    dest_path = DATASET2 + 'RD_JPG/' + f'{folder_name2[folder_index]}/images/'
    print(f"destination path:    {dest_path}")

    for images in os.listdir(src_path):
        # check if the images ends with .png
        if (images.endswith(".png")):
            count += 1
            print(count)
            # shutil.copyfile(src_path + images, dest_path + f'{count}.jpg')
            
    if count != 7193: print("Error!")



if __name__ == '__main__':
    tic = time.perf_counter()

    # delete_useless_files()
    # copy_and_rename_labels()
    copy_and_rename_labels2()
    # copy_and_rename_images()
    
    toc = time.perf_counter()
    duration = toc - tic
    print(f"duration: {duration:0.4f} seconds")

    # RD_256  duration: 18.6437 seconds
    # RD_416  duration: 20.0160 seconds
    # RD_YOLO duration: 19.3529 seconds


