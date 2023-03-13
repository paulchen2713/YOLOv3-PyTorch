# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:57:39 2022

@patch: 
    2022.08.01
    2023.02.17
@author: Paul
@file: train.py
@dependencies:
    env pt3.7
    python 3.7.13
    pytorch==1.7.1
    torchaudio==0.7.2
    torchvision==0.8.2
    tqdm==4.56.0

Main file for training YOLOv3 model on RD maps, Pascal VOC and COCO dataset
"""

import torch
import torch.optim as optim
from tqdm import tqdm # for progress bar

import config # for hyper-parameter tuning stuffs
from model import YOLOv3
from loss import YoloLoss
from utils import (
    mean_average_precision, 
    cells_to_bboxes, # convert the cells to actual bounding boxes relative to the entire image
    get_evaluation_bboxes, 
    save_checkpoint, 
    load_checkpoint, 
    check_class_accuracy, 
    get_loaders, 
    plot_couple_examples
)

torch.backends.cudnn.benchmark = True

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    # for batch_idx, (x, y) in enumerate(loop):
    # x, y = image, tuple(targets)
    for x, y in loop:
        # print(x.shape) # current shape: torch.Size([16, 416, 416, 3]), correct shape: torch.Size([16, 3, 416, 416])
        # x.permute(0, 3, 1, 2) # torch.Size([16, 416, 416, 3]) --> torch.Size([16, 3, 416, 416])

        # RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.HalfTensor) should be the same

        x = x.to(config.DEVICE)
        # y0, y1, y2 = (y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE), )
        y0, y1, y2 = y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE)

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (loss_fn(out[0], y0, scaled_anchors[0]) 
                  + loss_fn(out[1], y1, scaled_anchors[1]) 
                  + loss_fn(out[2], y2, scaled_anchors[2])) 

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)



def main():
    # RuntimeError: CUDA out of memory. Tried to allocate 22.00 MiB 
    # (GPU 0; 6.00 GiB total capacity; 5.27 GiB already allocated; 0 bytes free; 5.31 GiB reserved in total by PyTorch)
    # just change to smaller batch size, say 16.

    # references:
    # How to avoid "CUDA out of memory" in PyTorch (https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch)
    # xxx GiB reserved in total by PyTorch (https://blog.csdn.net/weixin_57234928/article/details/123556441)

    # torch.cuda.empty_cache() # doesn't work
    # del variables            # but there seems no variable to delete
    # gc.collect()
    # torch.cuda.memory_summary(device=None, abbreviated=False) # doesn't work

    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    # first test with "/8examples.csv" and "/100examples.csv" before moving on to "/train.csv" and "/test.csv"
    # train_loader, test_loader, train_eval_loader = get_loaders(
    train_loader, test_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
    )

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    scaled_anchors = (torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # plot_couple_examples(model=model, loader=test_loader, thresh=0.6, iou_thresh=0.5, anchors=scaled_anchors)

        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if config.SAVE_MODEL and epoch % 20 == 0 and epoch > 0:
            # for Pascal VOC Dataset, "D:/Datasets/PASCAL_VOC/checkpoint.pth.tar"
            # for RD_map Dataset, "D:/Datasets/RD_maps/checks/checkpoint.pth.tar"
            save_checkpoint(model, optimizer, filename=f"D:/Datasets/RD_maps/checks/checkpoint.pth.tar")

        print(f"Currently epoch {epoch}")
        print("On Train loader:")
        check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        # # train eval caused some errors
        # print("On Train Eval loader:")
        # check_class_accuracy(model, train_eval_loader, threshold=config.CONF_THRESHOLD)

        # just testing for 1 epoch
        print("On Test loader:")
        check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

        # pred_boxes, true_boxes = get_evaluation_bboxes(
        #     loader=test_loader,
        #     model=model,
        #     iou_threshold=config.NMS_IOU_THRESH,
        #     anchors=config.ANCHORS,
        #     threshold=config.CONF_THRESHOLD,
        # )
        # mapval = mean_average_precision(
        #     pred_boxes=pred_boxes,
        #     true_boxes=true_boxes,
        #     iou_threshold=config.MAP_IOU_THRESH,
        #     box_format="midpoint",
        #     num_classes=config.NUM_CLASSES,
        # )
        # print(f"mAP: {mapval.item()}")
        
        if epoch % 10 == 0 and epoch > 0:
            print("On Test loader:")
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)

            # it took around at least 10+ minutes to compute mAP
            pred_boxes, true_boxes = get_evaluation_bboxes(
                loader=test_loader,
                model=model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes=pred_boxes,
                true_boxes=true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"mAP: {mapval.item()}")



if __name__ == "__main__":
    main()

