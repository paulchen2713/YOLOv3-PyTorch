# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 18:30:27 2022

@author: Paul
@file: resize_images.py
@dependencies:
    env pt3.7
    python 3.7.13
    torch >= 1.7.1
    torchvision >= 0.8.2
    Pillow >= 8.1.0

Resize image to a certain size
"""
# import the required libraries
import torchvision.transforms as T # for resizing the images
from PIL import Image              # for loading and saving the images
import os
from os import listdir

# set the dataset path
DATASET = 'D:/Datasets/RADA/RD/'
DATASET2 = 'D:/Datasets/CARRADA2/'

CURR_PATH = 'D:/BeginnerPythonProjects/read_carrada/'

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


# test the basic functionality of resizing an image to certain size
def testing(i=1, file_type='jpg'):
    # read the input image
    # img = Image.open(f'D:/Datasets/RD_maps/scaled_colors/{i}_sc.png')
    img = Image.open(f'D:/Datasets/RD_maps/scaled_colors/{i}_sc.{file_type}')

    # compute the size (width, height) of image
    before = img.size
    print(f"original image size: {before}")

    # define the transform function to resize the image with given size, say 416-by-416
    transform = T.Resize(size=(416,416))

    # apply the transform on the input image
    img = transform(img)

    # check the size (width, height) of image
    after = img.size
    print(f"resized image size: {after}")

    # overwrite the original image with the resized one
    # img = img.save(f'D:/Datasets/RD_maps/scaled_colors/{i}_sc.png')
    img.show()


def main(max_iter=1, file_type='jpg'):
    # 1600
    for i in range(1, max_iter + 1):
        # read the input image
        img = Image.open(f'D:/Datasets/RD_maps/scaled_colors/{i}_sc.{file_type}')

        # define the transform function to resize the image with given size, say 416-by-416
        transform = T.Resize(size=(416,416))

        # apply the transform on the input image
        img = transform(img)

        # overwrite the original image with the resized one
        img = img.save(f'D:/Datasets/RD_maps/scaled_colors/{i}_sc.{file_type}')
        print(f"{i}")


def temp():
    name = ['rrdm', 'rd_matrix']
    for n in name:
        # read the input image
        img = Image.open(CURR_PATH + f'figs/{n}.png')

        # compute the size (width, height) of image
        before = img.size
        print(f"original image size: {before}")

        # define the transform function to resize the image with given size
        transform = T.Resize(size=(256, 64))

        # apply the transform on the input image
        img = transform(img)

        # check the size (width, height) of image
        after = img.size
        print(f"resized image size: {after}")

        # overwrite the original image with the resized one
        img = img.save(CURR_PATH + f'figs/resized/{n}.png')


def resize_to_64_256():
    count = 1
    for dir_name in dir_names: # [23:24]: # 
        print(f"current directory: {dir_name}")

        # set the file path
        seq_path = DATASET + dir_name + '/images/'
        print(f"current seq path: {seq_path}")

        for images in os.listdir(seq_path):
            # check if the image ends with png
            if (images.endswith(".png")):
                # print(count, seq_path + images)

                # read the input image
                img = Image.open(seq_path + images)

                # # compute the size (width, height) of image
                # before = img.size
                # print(f"original image size: {before}")

                # define the transform function to resize the image with given size
                transform = T.Resize(size=(256, 64))

                # apply the transform on the input image
                img = transform(img)

                # # check the size (width, height) of image
                # after = img.size
                # print(f"resized image size: {after}")

                # overwrite the original image with the resized one
                img = img.save(f'D:/Datasets/RADA/RD_all/images/{count}.png')
                
                print(count)
                count += 1



if __name__ == '__main__':
    # testing(1, 'jpg')
    # main(1600, 'jpg')
    resize_to_64_256()

    
