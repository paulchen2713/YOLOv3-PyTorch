# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:55:40 2023

@patch: 2023.06.25

@author: Paul
@file: evaluate.py
@dependencies:
    env pt3.8
    python==3.8.16
    numpy==1.23.5
    pytorch==1.13.1
    pytorch-cuda==11.7
    torchaudio==0.13.1
    torchvision==0.14.1
    pandas==1.5.2
    pillow==9.3.0
    tqdm==4.64.1
    albumentations==1.3.0
    matplotlib==3.6.2
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import os
import random
from datetime import date
import time

import config
from model import YOLOv3
from dataset import YOLODataset
from utils import mean_average_precision, get_evaluation_bboxes, load_checkpoint
from loss import YoloLoss

torch.backends.cudnn.benchmark = True



def test():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()

    test_csv_path = config.DATASET + "test.csv"
    test_dataset = YOLODataset(
        csv_file=test_csv_path,
        transform=config.test_transforms,
        S=config.S, 
        image_dir=config.IMAGE_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,  # NOTE has to be 1, originally was set as 'config.BATCH_SIZE'
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    # 'checkpoint-',
    checkpoint_value = [
        'checkpoint-2023-06-07-1',
        'checkpoint-2023-06-14-1',
        'checkpoint-2023-06-18-1',
        'checkpoint-2023-06-19-2',
        'checkpoint-2023-06-26-1',
        'checkpoint-2023-06-27-3',
        'checkpoint-2023-06-28-2',
        'checkpoint-2023-06-28-3',
        'checkpoint-2023-06-28-4',
        'checkpoint-2023-06-28-5',
    ]
    index = len(checkpoint_value) - 1
    checkpoint_file = f"{checkpoint_value[index]}.pth.tar"  
    load_checkpoint(
        config.DATASET + "checks/" + checkpoint_file,  # checkpoint-2023-05-22-2.pth.tar,  
        model, 
        optimizer, 
        config.LEARNING_RATE 
    )

    scaled_anchors = (torch.tensor(config.ANCHORS)* torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)

    check_file_name = f"{checkpoint_value[index]}"  # NOTE 
    print(f"Weights: {checkpoint_file}")
    print(f"Anchors: ")
    for _, anchor in enumerate(scaled_anchors):
        print(f"  {anchor}")
    print(f"")

    # file path for the testing statistics
    valid_path = config.DATASET + f'training_logs/valid/'
    # If the folder doesn't exist, then we create that folder 
    if os.path.isdir(valid_path) is False:
        print(f"creating 'valid' folder to store the stats")
        os.makedirs(valid_path)

    # write flags 
    write_all_stats, write_raw_data = True, True

    all_stats_file = valid_path + f"{check_file_name}-all_stats.txt"
    raw_data_file  = valid_path + f"{check_file_name}-raw_data.txt"
    if os.path.isfile(f"{all_stats_file}") is True:
        print(f"the '{check_file_name}-all_stats.txt' is already exits!")
        write_all_stats = False
    if os.path.isfile(f"{raw_data_file}") is True:
        print(f"the '{check_file_name}-raw_data.txt' is already exits!")
        write_raw_data = False
    
    losses = []
    # 1st tqdm progress bar
    if write_raw_data is True and write_raw_data is True:
        for _, (x, y) in enumerate(tqdm(test_loader)):
            x = x.to(config.DEVICE)
            y0, y1, y2 = y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE)

            out = model(x)
            
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

            losses.append(loss.item())
            optimizer.zero_grad()

            curr_class_preds, correct_class = 0, 0
            curr_noobj, correct_noobj = 0, 0
            curr_obj, correct_obj = 0, 0

            x = x.to(config.DEVICE)

            for i in range(3):
                y[i] = y[i].to(config.DEVICE)
                obj   = y[i][..., 0] == 1  # in paper this is Iobj_i
                noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

                correct_class += torch.sum(torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj])
                curr_class_preds += torch.sum(obj)

                obj_preds = torch.sigmoid(out[i][..., 0]) > config.CONF_THRESHOLD
                correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
                curr_obj += torch.sum(obj)

                correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
                curr_noobj += torch.sum(noobj)

            class_acc  = (correct_class / (curr_class_preds + 1e-16))*100
            no_obj_acc = (correct_noobj / (curr_noobj + 1e-16))*100
            obj_acc    = (correct_obj   / (curr_obj + 1e-16))*100
            
            if write_all_stats is True:
                with open(all_stats_file, "a") as txt_file:
                    print(f"loss value: {loss.item():0.4f},",        end="  ", file=txt_file)
                    print(f"class_accuracy: {class_acc:0.4f},",      end="  ", file=txt_file)
                    print(f"no_object_accuracy: {no_obj_acc:0.4f},", end="  ", file=txt_file)
                    print(f"object_accuracy: {obj_acc:0.4f}",        end="\n", file=txt_file)
            if write_raw_data is True:
                with open(raw_data_file, "a") as txt_file:
                    print(f"{loss.item():0.15f}, {class_acc:0.15f}, {no_obj_acc:0.15f}, {obj_acc:0.15f}", file=txt_file)
    
    # 
    pred_bbox_path = valid_path + f"{check_file_name}-pred_bbox.txt"
    true_bbox_path = valid_path + f"{check_file_name}-true_bbox.txt"
    if os.path.isfile(f"{pred_bbox_path}") and os.path.isfile(f"{true_bbox_path}"): 
        print(f"the '{check_file_name}-pred_bbox.txt' and '{check_file_name}-true_bbox.txt' are already exit!\n")
        return

    # 2nd tqdm progress bar
    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )

    with open(pred_bbox_path, "w") as txt_file:
        for bbox in pred_boxes:
            print(bbox, file=txt_file)
    with open(true_bbox_path, "w") as txt_file:
        for bbox in true_boxes:
            print(bbox, file=txt_file)

    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=config.NUM_CLASSES,
    )
    print(f"mAP: {mapval.item()}")



if __name__ == "__main__":

    tic = time.perf_counter()

    test()

    toc = time.perf_counter()
    duration = (toc - tic)
    print(f"duration:  {duration:0.4f} sec")

