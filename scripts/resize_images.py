# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 18:30:27 2022

@patch:
    2023.02.25
@author: Paul
@file: resize_images.py
@dependencies:
    env pt3.7 (my PC)
    python 3.7.13
    pytorch==1.7.1     py3.7_cuda110_cudnn8_0 pytorch
    torchaudio==0.7.2  py37 pytorch
    torchvision==0.8.2 py37_cu110 pytorch
    pillow==8.1.0

Resize image to a certain size
"""

# import the required libraries
import torchvision.transforms as T # for resizing the images
from PIL import Image              # for loading and saving the images
import os
from os import listdir
import time

# set the dataset path
DATASET = 'D:/Datasets/RADA/RD/'

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


def read_txt_file(file_name=""):
    dir_names = list()
    with open(DATASET + file_name, "r") as seqs_file:
        dir_names = seqs_file.read().splitlines()
    return dir_names
# e.g. read "validated_seqs.txt"
# temp = read_txt_file("validated_seqs.txt")


# test the basic functionality of resizing an image to certain size
def RD_maps_testing(i=1, file_type='jpg'):
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


def RD_maps_resizing(max_iter=1, file_type='jpg'):
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


def resize_to_n_by_n(n=64, debug_mode=False):
    print(f"Resizing every images to {n}-by-{n}")
    count = 0
    for dir_name in dir_names: # [23:24]: # 
        if debug_mode == True: print(f"current directory: {dir_name}")

        # set the file path
        seq_path = DATASET + dir_name + '/images/'
        if debug_mode == True: print(f"current seq path: {seq_path}")

        for images in os.listdir(seq_path):
            # check if the image ends with png
            if (images.endswith(".png")):
                if debug_mode == True: print(count, seq_path + images)

                # read the input image
                img = Image.open(seq_path + images)

                # compute the size (width, height) of image
                if debug_mode == True:
                    before = img.size
                    print(f"original image size: {before}")

                """
                The PyTorch method for resizing, named torchvision.transforms.Resize(), is a 
                wrapper around the PIL library, so that the results will be the same compared to Pillow.
                """
                # define the transform function to resize the image with given size
                transform = T.Resize(size=(n, n))

                # apply the transform on the input image
                img = transform(img)

                # check the size (width, height) of image
                if debug_mode == True:
                    after = img.size
                    print(f"resized image size: {after}")

                count += 1
                print(count)
                if debug_mode == True: 
                    print(f"store_path: D:/Datasets/RADA/RD_{n}/images/{count}.png")
                    break # under debug mode, we do not want to save the images

                # overwrite the original image with the resized one
                img = img.save(f'D:/Datasets/RADA/RD_{n}/images/{count}.png')
                


if __name__ == '__main__':
    tic = time.perf_counter()

    # testing(1, 'jpg')
    # main(1600, 'jpg')
    # resize_to_64_256()
    
    n = 416
    resize_to_n_by_n(n)
    print(f"Resizing every images to {n}-by-{n}")

    toc = time.perf_counter()
    duration = toc - tic
    print(f"duration: {duration:0.4f} seconds") 

    # Resizing every images to 256-by-256
    # duration: 176.5745 seconds

    # Resizing every images to 416-by-416
    # duration: 285.5628 seconds

