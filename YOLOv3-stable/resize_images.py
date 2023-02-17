# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 18:30:27 2022

@patch: 2022.08.01
@author: Paul
@file: resize_images.py
@dependencies:
    env pt3.7
    python 3.7.13
    pytorch==1.7.1
    torchaudio==0.7.2
    torchvision==0.8.2
    pillow==8.1.0 

Resize image to a certain size
"""
# import the required libraries
import torchvision.transforms as T # for resizing the images
from PIL import Image              # for loading and saving the images

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

if __name__ == '__main__':
    # testing(1, 'jpg')
    main(0, 'jpg')

