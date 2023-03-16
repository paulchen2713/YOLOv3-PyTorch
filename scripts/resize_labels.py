# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 10:05:35 2023

@patch:
    2023.02.26
@author: Paul
@file: resize_labels.py
@dependencies:
    env pt3.7 (my PC)
    python 3.7.13
    pytorch==1.7.1     py3.7_cuda110_cudnn8_0 pytorch
    torchaudio==0.7.2  py37 pytorch
    torchvision==0.8.2 py37_cu110 pytorch
    pillow==8.1.0

Resize the labels to fit the resized images to a certain size
"""

import json
import os
from os import listdir
import time
from datetime import date

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


def resize_to_n_by_n(out_shape=64, debug_mode=False, data_type='RDM', out_type='YOLO', store=False):
    print(f"Generating labels for {out_shape}-by-{out_shape} matrices")
    count = 0
    for dir_name in dir_names[23:24]: # : # 
        if debug_mode == True: print(f"current directory: {dir_name}")

        # set the file path
        file_index = ["range_doppler_light.json", "range_angle_light.json"]
        with open(f"D:/Datasets/CARRADA/{dir_name}/annotations/box/" + f"{file_index[0]}", "r") as json_file:
            # read out all the bbox labels 
            data = json.loads(json_file.read())
        
        # extract all keys from the dict, and store them in a list()
        all_keys = list(data.keys())

        for key in all_keys[62:63]: # : # 
            print(f"frame name: \"{key}\"")

            dest_dirs = ['RD_64', 'RD_256', 'RD_416']
            #  set the store path
            DEST_PATH = f"D:/Datasets/RADA/{dest_dirs[0]}/labels/"
            if debug_mode == True:
                print(f"num of boxes: {len(data[key]['boxes'])}")
                print(f"num of labels: {len(data[key]['labels'])}")

            if (len(data[key]['boxes']) != len(data[key]['labels'])): print("boxes and labels are mismatched!")

            count += 1
            with open(DEST_PATH + f"{count}.txt", "w") as label_txt_file:
                # in each rd_matrix / image it may contain 1~3 possible targets
                for index in range(0, len(data[key]['boxes'])):
                    class_index = data[key]['labels'][index] - 1
                    if debug_mode == True:
                        print(data[key]['boxes'][index])
                        print(data[key]['labels'][index])
                        print(f"class_index = {class_index}")
                    
                    # [x, y, width, height] is COCO format in absolute scale
                    # [x_min, y_min, x_max, y_max] is Pascal_VOC format in absolute scale
                    x_min, y_min, x_max, y_max = data[key]['boxes'][index][0:4]   # extract Pascal_VOC / COCO format in absolute scale

                    if out_shape == 64: 
                        x_min, x_max = x_min / 4, x_max / 4
                        if debug_mode == True: print(f"out_shape is {out_shape}, (x_min, x_max) = {x_min}, {x_max}")

                    if out_shape == 256: 
                        y_min, y_max = y_min * 4, y_max * 4
                        if debug_mode == True: print(f"out_shape is {out_shape}, (y_min, y_max) = {y_min}, {y_max}")

                    if debug_mode == True:
                        print(f"(class, x_min, y_min, x_max, y_max) = ({class_index} {x_min} {y_min} {x_max} {y_max})")

                    if out_type == 'YOLO':
                        """
                        make sure it's [class_id, x, y, width, height] in relative scale
                        """
                        x, y = (x_max + x_min) / 2, (y_max + y_min) / 2
                        w, h = (y_max - y_min), (x_max - x_min)
                        if data_type == 'RDM':
                            # convert resized RD map from COCO format to YOLO format in relative scale
                            x, y, w, h = x / out_shape, y / out_shape, w / out_shape, h / out_shape 
                        elif data_type == 'RAM':
                            x, y, w, h = x / 256, y / 256, w / 256, h / 256 # RA map
                        if debug_mode == True: print(f"(class, x, y, w, h) = ({class_index}, {x}, {y}, {w}, {h}) in relative scale")
                        # redirect 'print()' output to a file
                        if store == True: print(f"{class_index} {x} {y} {w} {h}", file=label_txt_file) 
                    elif out_type == 'COCO':
                        """
                        make sure it's [class_id, x, y, width, height] in absolute value
                        """
                        x, y = (x_max + x_min) / 2, (y_max + y_min) / 2
                        w, h = (y_max - y_min), (x_max - x_min)

                        if debug_mode == True:
                            print(f"(class, x, y, w, h) = ({class_index}, {x}, {y}, {w}, {h}) in absolute value")
                        
                        if store == True:
                            print(f"{class_index} {x} {y} {w} {h}", file=label_txt_file) # redirect 'print()' output to a file
                    elif out_type == 'Pascal_VOC':
                        """
                        make sure it's [class_id, x_min, y_min, x_max, y_max] in relative scale
                        """
                        if data_type == 'RDM':
                            x_min, y_min, x_max, y_max = x_min / 256, y_min / 64, x_max / 256, y_max / 64
                        elif data_type == 'RAM':
                            x_min, y_min, x_max, y_max = x_min / 256, y_min / 256, x_max / 256, y_max / 256
                        
                        if debug_mode == True:
                            print(f"(class, x_min, y_min, x_max, y_max) = ({class_index} {x_min} {y_min} {x_max} {y_max}) in relative scale")
                        
                        if store == True:
                            print(f"{class_index} {x_min} {y_min} {x_max} {y_max}", file=label_txt_file) # redirect 'print()' output to a file
                    # print("---------------------------")



def absolute_scale_xywh():
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

            folder = ['absolute_scale', 'relative_scale', 'absolute_xywh', 'relative_xywh']
            #  set the store path
            DEST_PATH = f"D:/Datasets/RADA/{folder[3]}/"
            if (len(data[key]['boxes']) != len(data[key]['labels'])): print("boxes and labels are mismatched!")

            count += 1
            print(count)
            with open(DEST_PATH + f"{count}.txt", "w") as label_txt_file:
                # in each rd_matrix / image it may contain 1~3 possible targets
                for index in range(0, len(data[key]['boxes'])):
                    class_index = data[key]['labels'][index] - 1
                    x_min, y_min, x_max, y_max = data[key]['boxes'][index][0:4]
                    if x_min < 0 or x_max > 256 or y_min < 0 or y_max > 64: 
                        print(f"image: {count}.txt, out of range!")
                        print(f"bbox: {class_index} {x_min} {y_min} {x_max} {y_max}")
                        break
                    # y_min, y_max = y_min * 4, y_max * 4
                    h = x_max - x_min
                    w = y_max - y_min
                    y = int((y_min + y_max) / 2)
                    x = int((x_min + x_max) / 2)

                    # rescale 
                    x, h = x / 256, h / 256
                    y, w = y / 64, w / 64
                    x_min, y_min, x_max, y_max = x_min / 256, y_min / 64, x_max / 256, y_max / 64
                    if x < 0 or x > 1 or y < 0 or y > 1 or w < 0 or w > 1 or h < 0 or h > 1:
                        print(f"image: {count}.txt, out of range!")
                        print(f"bbox: {class_index} {x} {y} {w} {h}")
                        break

                    # print(f"{class_index} {x_min} {y_min} {x_max} {y_max}", file=label_txt_file)
                    print(f"{class_index} {y} {x} {w} {h}", file=label_txt_file)




if __name__ == '__main__':
    tic = time.perf_counter()

    out_shape = 256
    # resize_to_n_by_n(out_shape=out_shape, debug_mode=True, data_type='RDM', out_type='YOLO', store=False)
    absolute_scale_xywh()
    # print(f"Resizing every labels to {n}-by-{n}")

    toc = time.perf_counter()
    duration = toc - tic
    print(f"duration: {duration:0.4f} seconds")
