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
from torch.utils.data import DataLoader

from model import YOLOv3
from dataset import YOLODataset
# from model_with_weights2 import YOLOv3
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

from tqdm import tqdm
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
log_file_name = '2023-07-12-2' # TODO date_function.today() 

# we are checking whether '<log_file_name>.txt' file exists in the 'losses' folder
file2check = config.DATASET + f'training_logs/train/losses/{log_file_name}.txt'  
# assert os.path.isfile(f"{file2check}") is False, f"the 'training_logs/train/losses/{log_file_name}.txt' file already exists!"

# isTesting = False  ## 
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
    # if isTesting is False:
    with open(loss_path + f"mean_loss/{log_file_name}.txt", "a") as loss_file:
        print(f"{mean_loss}", file=loss_file)
    # if isTesting is False:
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
            config.DATASET + f"checks/yolov3_carrada_47.55_map.pth.tar",  # config.CHECKPOINT_FILE, 
            model, 
            optimizer, 
            config.LEARNING_RATE 
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    # for updating mAP and determine whether should I save the model
    curr_mAP, maxi_mAP = 0, 0
    isBetter = False 

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors) # just plotting some images without bboxes
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        print(f"Currently on epoch {epoch}")
        print(f"On Train loader:")
        class_acc, no_obj_acc, obj_acc = check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        
        # file path for the training statistics
        file_path = config.DATASET + f'training_logs/train/'

        # store the class_acc, no_obj_acc, obj_acc values of every epoch to text files named as today's date
        # if isTesting is False:
        with open(file_path + f"class_accuracy/{log_file_name}.txt", "a") as txt_file:
            print(f"{class_acc}", file=txt_file)

        with open(file_path + f"no_object_accuracy/{log_file_name}.txt", "a") as txt_file:
            print(f"{no_obj_acc}", file=txt_file)

        with open(file_path + f"object_accuracy/{log_file_name}.txt", "a") as txt_file:
            print(f"{obj_acc}", file=txt_file)

        # config.TEST_POINT
        if epoch % 5 == 0 and epoch > 0:
            print(f"On Test loader (epoch {epoch}):")
            class_acc, no_obj_acc, obj_acc = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

            # file path for the testing statistics
            file_path = config.DATASET + f'training_logs/test/'

            # store the class_acc, no_obj_acc, obj_acc values of every epoch to text files named as today's date
            # if isTesting is False:
            with open(file_path + f"class_accuracy/{log_file_name}.txt", "a") as txt_file:
                print(f"{class_acc}", file=txt_file)

            with open(file_path + f"no_object_accuracy/{log_file_name}.txt", "a") as txt_file:
                print(f"{no_obj_acc}", file=txt_file)

            with open(file_path + f"object_accuracy/{log_file_name}.txt", "a") as txt_file:
                print(f"{obj_acc}", file=txt_file)

        # config.TEST_POINT
        if epoch % 5 == 0 and epoch > 0:
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
            curr_mAP = mapval.item()
            print(f"cur mAP:  {curr_mAP}")

            file_path = config.DATASET + f'training_logs/mAP/'
            # if isTesting is False:
            with open(file_path + f"{log_file_name}.txt", "a") as txt_file:
                print(f"{curr_mAP}", file=txt_file)

            if curr_mAP > maxi_mAP:
                maxi_mAP = curr_mAP
                isBetter = True
                print(f"max mAP:  {maxi_mAP}")
            elif curr_mAP <= maxi_mAP:
                isBetter = False
                print(f"max mAP:  {maxi_mAP}")
        # 
        if config.SAVE_MODEL and isBetter: 
            file_name = config.DATASET + f"checks/checkpoint-{log_file_name}.pth.tar"
            print(f"---> Saving checkpoint with max mAP:  {maxi_mAP}")
            save_checkpoint(model, optimizer, filename=file_name)
            isBetter = False
        print(f"")


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

    checkpoint_value = ['checkpoint-2023-06-18-1']  
    index = 0
    checkpoint_file = f"{checkpoint_value[index]}.pth.tar"  
    load_checkpoint(
        config.DATASET + "checks/" + checkpoint_file,  # checkpoint-2023-05-22-2.pth.tar,  
        model, 
        optimizer, 
        config.LEARNING_RATE 
    )

    scaled_anchors = (torch.tensor(config.ANCHORS)* torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)

    check_file_name = f"{checkpoint_value[index]}-1"  # NOTE 

    # file path for the testing statistics
    valid_path = config.DATASET + f'training_logs/valid/'
    # If the folder doesn't exist, then we create that folder 
    if os.path.isdir(valid_path) is False:
        print(f"creating 'valid' folder to store the stats")
        os.makedirs(valid_path)

    losses = []
    # 1st tqdm progress bar
    for _, (x, y) in enumerate(tqdm(test_loader)):
        x = x.to(config.DEVICE)
        y0, y1, y2 = y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE)

        out = model(x)
        # with torch.no_grad():
        #     out = model(x)

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

        file_name = valid_path + f"all_stats-{check_file_name}.txt"
        with open(file_name, "a") as txt_file:
            print(f"loss value: {loss.item():0.4f},",        end="  ", file=txt_file)
            print(f"class_accuracy: {class_acc:0.4f},",      end="  ", file=txt_file)
            print(f"no_object_accuracy: {no_obj_acc:0.4f},", end="  ", file=txt_file)
            print(f"object_accuracy: {obj_acc:0.4f}",        end="\n", file=txt_file)
        with open(valid_path + f"raw_data-{check_file_name}.txt", "a") as txt_file:
            print(f"{loss.item():0.15f}, {class_acc:0.15f}, {no_obj_acc:0.15f}, {obj_acc:0.15f}", file=txt_file)

    # 2nd tqdm progress bar
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

    # file_name = valid_path + f"mAP-{check_file_name}.txt"
    # with open(file_name, "a") as txt_file:
    #     print(f"original: {mapval.item()}, truncated: {mapval.item():0.4f}", file=txt_file)



if __name__ == "__main__":

    tic = time.perf_counter()

    main()


    # NOTE 8-fold
    # 2023-07-13-1  epoch: 100   duration:   hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.  ##split 3 + smaller model-
    # 2023-07-12-2  epoch: 100   duration:   hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.  ##split 2 + smaller model-
    # 2023-07-12-1  epoch: 100   duration:  4.2602 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4926  ##split 1 + smaller model-11
    # 2023-07-11-1  epoch: 100   duration:  4.3234 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4488  ##split 0 + smaller model-11


    # NOTE 7-fold
    # 2023-06-28-5  epoch: 100   duration:  4.2641 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4518  ##split 6 + smaller model-10
    # 2023-06-28-4  epoch: 100   duration:  6.4446 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4685  ##split 5 + smaller model-10
    # 2023-06-28-3  epoch: 100   duration:  5.0467 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4576  ##split 4 + smaller model-9
    # 2023-06-28-2  epoch: 100   duration:  4.2279 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4448  ##split 3 + smaller model-8
    # 2023-06-28-1  epoch: 100   duration:  6.3814 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4631  ##split 2 + smaller model-9
    # 2023-06-27-4  epoch: 100   duration:  6.3816 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4205  ##split 1 + smaller model-8
    # 2023-06-27-3  epoch: 100   duration:  4.2196 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4912  ##split 0 + smaller model-9

    
    # NOTE 5-fold 
    # 2023-06-27-2  epoch: 100   duration:  6.1737 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4250  ##split 4 + smaller model-9
    # 2023-06-27-1  epoch: 100   duration:  4.0348 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4437  ##split 3 + smaller model-8
    # 2023-06-26-2  epoch: 100   duration:  4.0543 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.3944  ##split 2 + smaller model-7
    # 2023-06-26-1  epoch: 100   duration:  6.1282 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.3873  ##split 1 + smaller model-6
    # 2023-06-19-2  epoch: 100   duration: 10.8377 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.3631  ##split 0 + smaller model-5


    # NOTE 4-fold 
    # 2023-06-19-1  epoch: 100   duration:  4.4134 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4701  ##split 3 
    # 2023-06-18-3  epoch: 100   duration:  4.2812 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4549  ##split 2 
    # 2023-06-18-2  epoch: 100   duration:  4.1333 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4518  ##split 1 
    # 2023-06-18-1  epoch: 100   duration:  4.1300 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4747  ##split 0 


    # NOTE 10-fold 
    # 2023-06-17-3  epoch: 100   duration:  4.6962 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4986  ##split 9 
    # 2023-06-17-2  epoch: 100   duration:  4.7984 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4960  ##split 8 
    # 2023-06-17-1  epoch: 100   duration:  4.7631 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4798  ##split 7 
    # 2023-06-16-2  epoch: 100   duration:  4.9729 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4614  ##split 6 
    # 2023-06-16-1  epoch: 100   duration:  4.8746 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4617  ##split 5 
    # 2023-06-15-2  epoch: 100   duration:  4.7963 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4854  ##split 4 
    # 2023-06-15-1  epoch: 100   duration:  4.9219 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4951  ##split 3 
    # 2023-06-14-1  epoch: 100   duration:  4.8926 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.5150  ##split 2 NOTE #1
    # 2023-06-13-2  epoch: 100   duration:  4.7045 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4868  ##split 1 
    # 2023-06-13-1  epoch: 100   duration:  5.0132 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4752  ##split 0 


    # NOTE 6-fold 
    # 2023-06-10-1  epoch: 100   duration:  4.5791 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4670  ##split 5 
    # 2023-06-09-1  epoch: 100   duration:  5.5375 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4551  ##split 4 
    # 2023-06-08-3  epoch: 100   duration:  4.7448 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4601  ##split 3 
    # 2023-06-08-2  epoch: 100   duration:  4.9901 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4786  ##split 2 
    # 2023-06-08-1  epoch: 100   duration:  4.6382 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4777  ##split 1 
    # 2023-06-07-1  epoch: 100   duration:  4.5955 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.5136  ##split 0 NOTE #1
    

    # 2023-06-05-1  epoch: 100   duration:  5.4366 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 14e-5  max mAP:  0.4688  ##split 0 
    # 2023-05-22-2  epoch: 100   duration:  4.0113 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 14e-5  max mAP:  0.3781 
    # 2023-05-22-1  epoch: 100   duration:  4.7716 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4138 

    # 2023-05-21-1  epoch: 400   duration: 19.1984 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 3e-4   max mAP:  0.4292
    # 2023-05-20-2  epoch: 200   duration: 12.1146 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 3e-4   max mAP:  0.4593 NOTE #3
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
    # 2023-05-11-1  epoch: 200   duration: 13.2833 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 15e-5  max mAP:  0.4755 NOTE #1
    # 2023-05-09-1  epoch: 300   duration: 22.8276 hours  WEIGHT_DECAY = 1e-4  LEARNING_RATE = 14e-5  max mAP:  0.4652 NOTE #2

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
    print(f"{log_file_name}  epoch: {config.NUM_EPOCHS}   duration:  {duration:0.4f} hours\n")
    # print(f"WEIGHT_DECAY = {config.WEIGHT_DECAY}  LEARNING_RATE = {config.LEARNING_RATE}  max mAP:  ")



