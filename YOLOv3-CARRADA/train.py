# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:51:47 2023

@patch: 
    2023.02.17
    2023.03.22

@author: Paul
@file: train.py
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

Main file for training YOLOv3 model on RD maps, Pascal VOC and COCO dataset
"""

import config
import torch
import torch.optim as optim

from model import YOLOv3
# from model_with_weights2 import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
from datetime import date as date_function
import time

torch.backends.cudnn.benchmark = True

import numpy as np
import os
import random


def seed_everything(seed=33):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# seed_everything()  # If you want deterministic behavior


# Using a unified 'log_file_name' for all file objects is necessary because if the training process runs across several days, 
# the log messages for the same training will be split into several files with different dates as their file names. However, 
# they actually belong in the same file. All log files will be named as the start date of the training.
log_file_name = '2023-05-21-1' # date_function.today()

# we are checking whether '<log_file_name>.txt' file exists in the 'losses' folder
file2check = config.DATASET + f'training_logs/train/losses/{log_file_name}.txt'  
# assert os.path.isfile(f"{file2check}") is False, f"the 'training_logs/train/losses/{log_file_name}.txt' file already exists!"


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for _, (x, y) in enumerate(loop):
        # print(x.shape) # current shape: torch.Size([16, 416, 416, 3]), correct shape: torch.Size([16, 3, 416, 416])
        # x.permute(0, 3, 1, 2) # torch.Size([16, 416, 416, 3]) --> torch.Size([16, 3, 416, 416])

        # RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.HalfTensor) should be the same

        x = x.to(config.DEVICE)
        # y0, y1, y2 = (y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE),)
        y0, y1, y2 = y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE)

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)
    
    loss_path = config.DATASET + f'training_logs/train/'
    # store the mean_loss value of every epoch to a text file named as today's date
    with open(loss_path + f"mean_loss/{log_file_name}.txt", "a") as loss_file:
        print(f"{mean_loss}", file=loss_file)
    
    for i_loss in losses:
        with open(loss_path + f"losses/{log_file_name}.txt", "a") as loss_file:
            print(f"{i_loss}", file=loss_file)




def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    # first test with "/8examples.csv" and "/100examples.csv" before moving on to "/train.csv" and "/test.csv"
    # train_loader, test_loader, train_eval_loader = get_loaders(
    train_loader, test_loader = get_loaders(
        train_csv_path=config.DATASET + "train.csv", test_csv_path=config.DATASET + "test.csv"
    )

    if config.LOAD_MODEL:
        load_checkpoint(
            config.DATASET + f"checks/yolov3_carrada_45.21_map.pth.tar",  # config.CHECKPOINT_FILE, 
            model, 
            optimizer, 
            config.LEARNING_RATE 
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors) # just plotting some images without bboxes
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        print(f"Currently epoch {epoch}")
        print("On Train loader:")
        class_acc, no_obj_acc, obj_acc = check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        
        # file path for the training statistics
        file_path = config.DATASET + f'training_logs/train/'

        # store the class_acc, no_obj_acc, obj_acc values of every epoch to text files named as today's date
        with open(file_path + f"class_accuracy/{log_file_name}.txt", "a") as txt_file:
            print(f"{class_acc}", file=txt_file)

        with open(file_path + f"no_object_accuracy/{log_file_name}.txt", "a") as txt_file:
            print(f"{no_obj_acc}", file=txt_file)

        with open(file_path + f"object_accuracy/{log_file_name}.txt", "a") as txt_file:
            print(f"{obj_acc}", file=txt_file)

        test_point = 10
        if epoch % config.TEST_POINT == 0 and epoch > 0:
            print("On Test loader:")
            class_acc, no_obj_acc, obj_acc = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

            # file path for the testing statistics
            file_path = config.DATASET + f'training_logs/test/'

            # store the class_acc, no_obj_acc, obj_acc values of every epoch to text files named as today's date
            with open(file_path + f"class_accuracy/{log_file_name}.txt", "a") as txt_file:
                print(f"{class_acc}", file=txt_file)

            with open(file_path + f"no_object_accuracy/{log_file_name}.txt", "a") as txt_file:
                print(f"{no_obj_acc}", file=txt_file)

            with open(file_path + f"object_accuracy/{log_file_name}.txt", "a") as txt_file:
                print(f"{obj_acc}", file=txt_file)

            # 
            if config.SAVE_MODEL:
                file_name = config.DATASET + f"checks/checkpoint-{log_file_name}.pth.tar"
                save_checkpoint(model, optimizer, filename=file_name)


        check_map = 10
        if epoch % config.TEST_POINT == 0 and epoch > 0:
            # 
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"mAP: {mapval.item()}")

            file_path = config.DATASET + f'training_logs/mAP/'
            with open(file_path + f"{log_file_name}.txt", "a") as txt_file:
                print(f"{mapval.item()}", file=txt_file)



if __name__ == "__main__":

    tic = time.perf_counter()

    main()

    # 2023-05-21-1  epoch: 400   duration:  hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 3e-4   
    # 2023-05-20-2  epoch: 200   duration: 12.1146 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 3e-4   max mAP:  0.4593
    # 2023-05-20-1  epoch:  80   duration:  3.8367 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 3e-4   max mAP:  0.4447
    # 2023-05-19-2  epoch: 150   duration: 12.5194 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 3e-4   max mAP:  0.4420

    # 2023-05-19-1  epoch: 150   duration: 12.9126 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 14e-5  max mAP:  0.4437
    # 2023-05-18-3  epoch:  80   duration:  5.7503 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 14e-5  max mAP:  0.4291
    # 2023-05-18-2  epoch:  80   duration:  5.1906 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 14e-5  max mAP:  0.4450
    # 2023-05-18-1  epoch:  80   duration:  5.5654 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4342

    # 2023-05-17-2  epoch:  80   duration:  5.7284 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 28e-5  max mAP:  0.4342
    # 2023-05-17-1  epoch:  80   duration:  5.5801 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 27e-5  max mAP:  0.4362
    # 2023-05-16-4  epoch:  80   duration:  5.3737 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 26e-5  max mAP:  0.4380
    # 2023-05-16-3  epoch:  80   duration:  4.3383 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 25e-5  max mAP:  0.4227
    
    # 2023-05-16-2  epoch:  80   duration:  5.3471 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 24e-5  max mAP:  0.4342
    # 2023-05-16-1  epoch:  80   duration:  5.6804 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 23e-5  max mAP:  0.4398
    # 2023-05-15-1  epoch:  80   duration:  5.6858 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 22e-5  max mAP:  0.4250
    # 2023-05-14-1  epoch:  80   duration:  4.0699 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 21e-5  max mAP:  0.4411
    
    # 2023-05-13-1  epoch:  80   duration:  5.6767 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 17e-5  max mAP:  0.4392
    # 2023-05-12-1  epoch:  80   duration:  7.0051 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 16e-5  max mAP:  0.4451
    # 2023-05-11-1  epoch: 200   duration: 13.2833 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4755
    # 2023-05-09-1  epoch: 300   duration: 22.8276 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 14e-5  max mAP:  0.4652

    # 2023-05-08-1  epoch: 200   duration: 14.2820 hours  WEIGHT_DECAY = 1e-3  LEARNING_RATE = 15e-5  max mAP:  0.4210
    # 2023-05-07-1  epoch: 100   duration:  8.3639 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 14e-5  max mAP:  0.4435
    # 2023-05-06-1  epoch: 100   duration:  8.7493 hours  WEIGHT_DECAY = 1e-3  LEARNING_RATE = 14e-5  max mAP:  0.4469
    # 2023-05-05-1  epoch: 400   duration: 26.4082 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4396 
    
    # 2023-05-04-2  epoch: 100   duration:  8.4733 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 20e-5  max mAP:  0.4227
    # 2023-05-04-1  epoch: 100   duration:  7.4228 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 19e-5  max mAP:  0.4431
    # 2023-05-03-2  epoch: 100   duration:  7.1676 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 18e-5  max mAP:  0.4468
    # 2023-05-03-1  epoch: 100   duration:  7.1341 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 17e-5  max mAP:  0.4469

    # 2023-05-02-4  epoch: 100   duration:  8.4758 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 16e-5  max mAP:  0.4342
    # 2023-05-02-3  epoch: 100   duration:  5.5819 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4520
    # 2023-05-02-2  epoch: 100   duration:  5.8219 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 14e-5  max mAP:  0.4521
    # 2023-05-02-1  epoch: 100   duration:  6.8200 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 13e-5  max mAP:  0.4374

    # 2023-05-01-3  epoch: 100   duration:  7.1689 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 12e-5  max mAP:  0.4372
    # 2023-05-01-2  epoch: 100   duration:  7.0366 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 11e-5  max mAP:  0.4490
    # 2023-05-01-1  epoch: 100   duration:  5.7350 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 10e-5  max mAP:  0.4386
    # 2023-04-30-2  epoch: 100   duration:  5.5800 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 9e-5   max mAP:  0.4356

    # 2023-04-30-1  epoch: 100   duration:  6.7780 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 8e-5   max mAP:  0.4340
    # 2023-04-29-2  epoch: 100   duration:  7.1015 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 7e-5   max mAP:  0.4309
    # 2023-04-29-1  epoch: 100   duration:  7.0542 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 6e-5   max mAP:  0.4267
    # 2023-04-28-3  epoch: 100   duration:  7.1785 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 5e-5   max mAP:  0.4143

    # 2023-04-28-2  epoch: 100   duration:  8.1383 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 4e-5   max mAP:  0.3963
    # 2023-04-22    epoch: 100   duration:  7.2117 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 3e-5   max mAP:  0.3792
    # 2023-04-27-2  epoch: 100   duration:  7.2838 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 2e-5   max mAP:  0.3233
    # 2023-04-28    epoch: 100   duration:  7.5511 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 1e-5   max mAP:  0.2697
    
    # 2023-04-27    epoch: 100   duration:  7.1676 hours  WEIGHT_DECAY = 1e-1  LEARNING_RATE = 3e-5   max mAP:  0.3289
    # 2023-04-26    epoch: 100   duration:  7.7900 hours  WEIGHT_DECAY = 1e-2  LEARNING_RATE = 3e-5   max mAP:  0.3646
    # 2023-04-25    epoch: 100   duration:  6.2753 hours  WEIGHT_DECAY = 1e-3  LEARNING_RATE = 3e-5   max mAP:  0.3603
    # 2023-04-22    epoch: 100   duration:  7.2117 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 3e-5   max mAP:  0.3792 

    # 2023-04-23    epoch: 300   duration: 20.8263 hours                                                                              max mAP:  0.4179
    # 2023-04-22    epoch: 100   duration:  7.2117 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 3e-5  'k_means() anchor'  'Shuffled'   max mAP:  0.3792
    # 2023-04-16    epoch: 100   duration:  8.0922 hours                                             'MiniBatchKMeans'                max mAP:  0.1736
    # 2023-04-15    epoch: 100   duration:  6.6698 hours                                             'KMeans() anchor'                max mAP:  0.1628
    # 2023-04-07    epoch: 1000  duration: 80.3616 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 1e-4  'YOLOv3 anchor'     'Serialized' max mAP:  0.1819

    toc = time.perf_counter()
    duration = (toc - tic) / 3600
    print(f"{log_file_name}  epoch: {config.NUM_EPOCHS}   duration:  {duration:0.4f} hours")
    # print(f"WEIGHT_DECAY = {config.WEIGHT_DECAY}  LEARNING_RATE = {config.LEARNING_RATE}  max mAP:  ")



