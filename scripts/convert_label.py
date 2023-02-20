# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:17:12 2022

@patch: 
    2022.10.30
    2022.12.20
@author: Paul
@file: convert_label.py
@dependencies:
    envs        pt3.7
"""

import json
import time

# set the dataset path
DATASET = 'D:/Datasets/CARRADA/'
DATASET2 = 'D:/Datasets/CARRADA2/'

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

def main():
    dir_names = read_txt_file("validated_seqs.txt")

    for dir_name in dir_names: # [23:24]: # 
        # e.g. "D:/Datasets/CARRADA/2019-09-16-12-58-42/annotations/box/"
        print(f"current directory: {dir_name}")

        # "range_doppler_light.json", "range_angle_light.json"
        file_index = ["range_doppler_light.json", "range_angle_light.json"]
        with open(DATASET + f"{dir_name}/annotations/box/" + f"{file_index[0]}", "r") as json_file:
        # with open(CURR_PATH + "range_doppler_light.json", "r") as json_file:
            """
            # there are two ways to read json file
            # data = json.load(json_file)         # one using json.load(), which load the json_file
            # data = json.loads(json_file.read()) # make sure you add ".read()" when using json.loads(), the "s" means string
            """
            data = json.loads(json_file.read())
            # print(type(data)) # <class 'dict'>

        # extract all keys from the dict, and store them in a list()
        all_keys = list(data.keys())
        # print(data[f"{all_keys[0]}"]["boxes"])  # [[69, 32, 72, 35]] <class 'list'>
        # print(data[f"{all_keys[0]}"]["labels"]) # [1] <class 'list'>
        
        for key in all_keys: # [62:63]: # 
            print(f"frame name: \"{key}\"")

            # paths of RD_maps and RA_maps
            RDM_PATH = f"D:/Datasets/CARRADA2/RD_Pascal_VOC/{dir_name}/labels/" # path that we store our labels
            RAM_PATH = f"D:/Datasets/CARRADA2/RA/{dir_name}/labels/" # path that we store our labels

            # we have to set 2 different paths of 'RDM_PATH' or 'RAM_PATH',
            # 2 different data types of 'RDM' or 'RAM', and
            # 3 different output annotation types of 'Pascal_VOC', 'COCO' or 'YOLO'
            def store_labels(data_path=f'{RDM_PATH}', data_type='RDM', out_type='YOLO', mode='', store=True):
                with open(data_path + f"0000_log_{dir_name}.txt", "a") as label_txt_file:
                    print(f"{data[key]}", file=label_txt_file)
                if mode == 'debug':
                    print(f"num of boxes: {len(data[key]['boxes'])}")
                    print(f"num of labels: {len(data[key]['labels'])}")

                if (len(data[key]['boxes']) != len(data[key]['labels'])): 
                    print("boxes and labels are mismatched!")

                with open(data_path + f"{key}.txt", "w") as label_txt_file:
                    # in each rd_matrix / image it may contain 1~3 possible targets
                    for index in range(0, len(data[key]['boxes'])):
                        class_index = data[key]['labels'][index] - 1
                        if mode == 'debug':
                            print(data[key]['boxes'][index])
                            print(data[key]['labels'][index])
                            print(f"class_index = {class_index}")
                        
                        # [x, y, width, height] is COCO format in absolute scale
                        # [x_min, y_min, x_max, y_max] is Pascal_VOC format in absolute scale
                        x_min, y_min, x_max, y_max = data[key]['boxes'][index][0:4]   # extract Pascal_VOC / COCO format in absolute scale
                        if mode == 'debug':
                            print(f"(class, x_min, y_min, x_max, y_max) = ({class_index} {x_min} {y_min} {x_max} {y_max})")

                        if out_type == 'YOLO':
                            """
                            make sure it's [class_id, x, y, width, height] in relative scale
                            """
                            x, y = (x_max + x_min) / 2, (y_max + y_min) / 2
                            w, h = (y_max - y_min), (x_max - x_min)
                            if data_type == 'RDM':
                                x, y, w, h = x / 256, y / 64, w / 64, h / 256 # RD map, convert from COCO format to YOLO format in relative scale
                            elif data_type == 'RAM':
                                x, y, w, h = x / 256, y / 256, w / 256, h / 256 # RA map
                            
                            if mode == 'debug':
                                print(f"(class, x, y, w, h) = ({class_index}, {x}, {y}, {w}, {h}) in relative scale")

                            if store == True:
                                print(f"{class_index} {x} {y} {w} {h}", file=label_txt_file) # redirect 'print()' output to a file
                        elif out_type == 'COCO':
                            """
                            make sure it's [class_id, x, y, width, height] in absolute value
                            """
                            x, y = (x_max + x_min) / 2, (y_max + y_min) / 2
                            w, h = (y_max - y_min), (x_max - x_min)

                            if mode == 'debug':
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
                            
                            if mode == 'debug':
                                print(f"(class, x_min, y_min, x_max, y_max) = ({class_index} {x_min} {y_min} {x_max} {y_max}) in relative scale")
                            
                            if store == True:
                                print(f"{class_index} {x_min} {y_min} {x_max} {y_max}", file=label_txt_file) # redirect 'print()' output to a file
                        # print("---------------------------")
            
            # call out store_labels function to extract the labels that we need
            store_labels(
                data_path=RDM_PATH,     # data_path=RDM_PATH or RAM_PATH
                data_type='RDM',        # data_type='RDM' or 'RAM'
                out_type='Pascal_VOC',  # out_type='YOLO', 'COCO' or 'Pascal_VOC'
                mode='',                # mode='debug' # means print everything out
                store=True              # store=True # means renew the .txt label, visa vera
            ) 
            


if __name__ == '__main__':
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    duration = toc - tic
    print(f"duration: {duration:0.4f} seconds") 
    print(duration)
    # rd_matrix duration: 4.6774586 seconds
    # ra_matrix duration: 4.3379489 seconds
    




