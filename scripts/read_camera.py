# -*- coding: utf-8 -*-
"""
Created on Sat Jun 03 18:40:03 2023

@patch: 2023.06.03
@author: Paul
@file: read_camera.py
@dependencies:
    env pt3.8
    python==3.8.16
    numpy==1.23.5
    pytorch==1.13.1
    pytorch-cuda==11.7
    torchaudio==0.13.1
    torchvision==0.14.1
    matplotlib==3.6.2
"""

from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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

# number of images / labels in each directory, total number of labels: 7193
num_of_images = [286, 273, 304, 327, 218, 219, 150, 208, 152, 174, 
                 174, 235, 442, 493, 656, 523, 350, 340, 304, 108, 
                 129, 137, 171, 143, 104, 81, 149, 124, 121, 98]


def main(debug_mode=False):
    count = 0
    for dir_name in dir_names: # [23:24]: # 
        # e.g. "D:/Datasets/CARRADA/2020-02-28-13-09-58/annotations/box/"
        if debug_mode == True: print(f"current directory: {dir_name}")

        # set the file path
        dir_path = DATASET + dir_name + '/'
        if debug_mode == True: print(f"current directory path: {dir_path}")

        # "range_doppler_light.json", "range_angle_light.json"
        with open(DATASET + f"{dir_name}/annotations/box/" + "range_doppler_light.json", "r") as json_file:
            data = json.loads(json_file.read())
        
        # extract all keys from the dict, and store them in a list()
        all_keys = list(data.keys())
        for key in all_keys: # [62:63]: # 
            if debug_mode == True: print(f"frame name: \"{key}\"")

            # set image path
            image_path = dir_path + 'camera_images/' + key + '.jpg'
            curr_image = Image.open(image_path)
            # curr_image.show()   

            store_folder = ['images', 'mats', 'camera_images']
            store_path = f"D:/Datasets/RADA/RD_JPG/{store_folder[2]}/" 

            # e.g. "D:/Datasets/CARRADA/2020-02-28-13-09-58/RD_maps/images/""
            if debug_mode == True: print(f"store path: \"{store_path}\"") 
            
            count += 1
            print(count)
            new_name = store_path + f'{count}.jpg'
            curr_image.save(new_name)



if __name__ == "__main__":
    tic = time.perf_counter()
    
    # main(debug_mode=False)  # duration: 276.6079 seconds

    toc = time.perf_counter()
    duration = toc - tic
    print(f"duration: {duration:0.4f} seconds")

