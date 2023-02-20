# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:09:11 2022

@author: Paul
@file: iterate_all_files.py
@dependencies:
    env pt3.7
    python 3.7.13
"""

# import the required libraries
import torchvision.transforms as T # for resizing the images
from PIL import Image              # for loading and saving the images
import os
from os import listdir
import shutil

folder_name = ['RD_Pascal_VOC', 'RD_YOLO', 'RD_COCO', 'RD', 'RD2', 'RD3']

# set the dataset path
DATASET = f'D:/Datasets/CARRADA2/{folder_name[1]}/'

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




if __name__ == '__main__':
    # delete_useless_files()
    copy_and_rename_labels()
    


