# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 19:27:14 2023

@patch: 
    2023.02.16
    2023.03.22

@author: Paul
@file: dataset.py
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

@references:
    Redmon, Joseph and Farhadi, Ali, YOLOv3: An Incremental Improvement, April 8, 2018. (https://doi.org/10.48550/arXiv.1804.02767)
    Ayoosh Kathuria, Whats new in YOLO v3?, April, 23, 2018. (https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)
    Sanna Persson, YOLOv3 from Scratch, Mar 21, 2021. (https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0)

Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

# set to Ture so that we don't get any errors when we loading the images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# create our own custom YOLODataset, first we're going to inherit from torch.utils.data.Dataset
# in a PyTorch Dataset there are 3 building blocks: __init__, __len__, and __getitem__, total 3 methods

# we will load an image and its bounding boxes and perform augmentations on both, for each bounding box we will then 
# assign it to the grid cell which contains its midpoint and decide which anchor is responsible for it by determining 
# which anchor the bounding box has highest iou with
class YOLODataset(Dataset):
    def __init__(self, 
        csv_file, 
        image_dir, 
        label_dir, 
        anchors, 
        image_size=416, # image_size=416, 
        S=[13, 26, 52], # stride size S=[13, 26, 52],
        C=20, # number of classes C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file) # "D:/Datasets/PASCAL_VOC/train.csv"
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size #
        self.transform = transform   # transform
        self.S = S                   # a list of grid sizes, S = [image_size // 32, image_size // 16, image_size // 8]
        # put all the anchors together for all 3 scales, each scale have 3 anchors stored in anchors[i]
        # combine the list below to a tensor of shape (9,2)
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) 
        self.num_anchors = self.anchors.shape[0] # 9
        self.num_anchors_per_scale = self.num_anchors // 3 # divided by 3, because we're assuming that we have 3 scales
        self.C = C                   # number of classes
        self.ignore_iou_thresh = 0.5 # if the iou is greater than 0.5, then we're going to ignore the prediction for that one

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # print(f"index: {index}")
        # get the index-th data, in the csv files, data are structured as index.jpg,index.txt, 
        # so (indxe, 0) get us the image and (index, 1) get us the label

        # get the label directory path (self.label_dir), then get the csv file name (self.annotations), then get the .txt file 
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # on the second column (which is 1)
        # print(label_path) # e.g. D:/Datasets/PASCAL_VOC/labels/003077.txt

        # after getting the .txt file path, we then load the .txt file which is delimited by space, and we set the returned array 
        # will have at least ndmin=2 dimensions, the original label is [class, x, y, w, h]

        # later on, we're going to use the albumentations library to augment both the image and bounding boxes, but it 
        # requires the class label to be last in the bounding box, which is [x, y, w, h, class], we use np.roll() to do so
        # np.roll(array, shift=4, axis=1) will roll the array to the right by 4 on the x-axis (axis=1 means horizontally)
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), shift=4, axis=1).tolist()

        # get the image directory path (self.image_dir), then get the csv file name (self.annotations), then get the .jpg file
        img_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0]) # on the first column (which is 0)
        # after getting the .jpg file path, we then load the .jpg file and we're going to convert it into RGB, for using albumentations 
        # library, we need to also make sure that it's a np array, it requires the image and bounding boxes both to be numpy arrays
        image = np.array(Image.open(img_path).convert("RGB"))

        # check if we have some transforms
        if self.transform:
            # send in the image and bounding boxes that way if we rotate the image, the bounding boxes are still gonna be correct
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # we make two assumptions which are there is only one label per bounding box and there is an equal number of 
        # bounding boxes (anchors) on each scale, which is 3 scale predictions (as paper)
        # each target for a particular scale and image will have shape (num of anchors // 3, grid_size, grid_size, 6)

        # self.num_anchors // 3 = 9 // 3 = 3 = num of scale predictions, S = grid_size for each scale
        # where 6 corresponds to [P(Object), x, y, w, h, class_label] which is in total 6 values
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        # print("len(targets): ", len(targets)) # len(targets): 3

        # then we loop through all the bounding boxes in this particular image (expensive step), each of the 3 scales should 
        # do a prediction and we need to assign which anchor should be responsible at which particular cells
        for box in bboxes:
            # each box contain [x, y, w, h, class], e.g. box = (0.5, 0.625, 0.125, 0.125, 0.0)  
            # now we have the targets for all the different scales, we want to go through each of our bounding box in the image
            # and we want to assign which anchor should be responsible , which cell should be responsible for all 3 diff scales

            # so how we assign which anchor is responsible? we check which one has the highest iou
            # send in the target's [width, height] and the pre-defined anchors (self.anchors) to calculate the ious for the 
            # particular box and for all the anchors, which are 9 diff ones
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) 

            # then we want to check which anchor is the best? we use torch.argsort() along y-axis (dim=0) with descending order, 
            # such that the index of the anchor with the largest iou with the target box appears first in the list.
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) # so the first one, first indicies is the best anchor

            x, y, width, height, class_label = box  # we can also take out the [x, y, width, height] from the box
            has_anchor = [False, False, False]      # we're going to make sure that each of 3 scales should have 1 anchor

            # then we will then loop through the 9 indices to assign the target to the best anchors, from best to worse 
            # our goal is to assign each target bounding box to an anchor on each scale i.e. in total assign each target 
            # to 1 anchor in each of the target matrices "targets" that we intialized above
            for anchor_idx in anchor_indices:
                # e.g. anchor_index = 0~8

                # how we check which scale it belongs to? 
                # scale_idx should be 0, 1, or 2, indicates which target we need to take out from the list of targets that we have above
                # NOTE when running under python 3.7 env, we can and have to use '//' 
                scale_idx = anchor_idx // self.num_anchors_per_scale # which scale

                # UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch. 
                # To keep the current behavior, use torch.div(a, b, rounding_mode='trunc'), or for actual floor division, 
                # use torch.div(a, b, rounding_mode='floor') # https://pytorch.org/docs/stable/generated/torch.div.html
                # NOTE when running under python 3.8 or above env, we have to use 'div()' 
                # scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='floor')

                # we also want to know which anchor on this particular scale are we assigning it to? 
                # anchor_on_scale should also be 0, 1, or 2, indicates which anchor in that particular scale that we want to use 
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # which anchor 

                S = self.S[scale_idx] # then we're going to get the grid size 

                # how many cells there are in this particular scale?
                # (y, x) are relative coordinates between [0, 1] and we want to get the absolute coordinates (i, j) in the image
                # i tells us which y cell, and j tells us which x cell
                i, j = int(S * y), int(S * x) # e.g. x = 0.5, S = 13 --> int(6.5) = 6
                # e.g. y, x = 0.625, 0.5 (should be the same as the box's y, x above)
                # e.g. i, j = 1, 1

                # the anchor taken is the one that take out from the scale_idx, then take out the specific anchor on this scale
                # targets[scale_idx] is checking in the list of diff target tensors, here we're taking out which anchor on that 
                # particular scale, then we're taking out the i, j for the particular cells, and we're taking out 0 for P(Object)

                # 
                #   File "d:\BeginnerPythonProjects\YOLOv3-PyTorch\dataset.py", line 163, in __getitem__
                #     anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] # e.g. originally tensor(0.)
                # IndexError: index 13 is out of bounds for dimension 2 with size 13
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] # e.g. originally tensor(0.)

                # it might be the case that this anchor is already been taken by another object, but it's super rare that you have 
                # two same object with the same bounding box, we want to make sure that we have not taken this before

                # we need to make sure that the anchor has not been taken, and we do not already have an anchor on this particular 
                # scale for this bounding box, because we want to have a prediction at each of the 3 scales
                if not anchor_taken and not has_anchor[scale_idx]:
                    # set the probability that there's an object, the object score P(Object) to 1, means there's an object
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    # the x, y values in this cell, both between [0,1]
                    x_cell, y_cell = S * x - j, S * y - i           # e.g. x_cell, y_cell = 0.0, 0.25 are <class 'float'> type

                    # the width and the height values in this cell, can be greater than 1 since it's relative to cell
                    # width_cell, height_cell = (width * S, height * S, )  
                    width_cell, height_cell = width * S, height * S # e.g. width_cell, height_cell = 0.5, 0.5 are <class 'numpy.float64'> type

                    # combine [x_cell, y_cell, width_cell, height_cell] as a list then convert it to a tensor
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell]) 
                    # e.g. box_coordinates = tensor([0.0000, 0.5000, 0.5000, 0.5000], dtype=torch.float64)

                    # NOTE. targets[scale_idx][anchor_on_scale, i, j, :] = [P(Object), x, y, w, h, class_label]
                    # read as: the targets of this particular scale index of this particular anchor on this scale of the (i, j) cell

                    # set 1:5 to box_coordinates, which is actually index 1, 2, 3, 4 which is [x, y, w, h]
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates 
                    # set 5 to int(class_label), because previously we load the label from box as 'float', but it's actually 'int'
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) 
                    # set the has_anchor flag of this particular scale index to True
                    has_anchor[scale_idx] = True 

                # if the anchor box is not taken and if the iou of a particulat bounding box is greater than the ignore threshold, 
                # then we're gonna ignore this prediction, by setting the P(Object) to -1
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore prediction

        # print(f"image shape: {image.shape}") # NOTE (416, 416, 3)
        # print(f"image type: {type(image)}")  # NOTE <class 'numpy.ndarray'> ??
        # in the end, we're gonna return the <class 'torch.Tensor'> type image?? and <class 'tuple'> type targets
        return image, tuple(targets)



def test():
    anchors = config.ANCHORS
    transform = config.test_transforms # None # 

    # S = [13, 26, 52]
    S = config.S 

    # PASCAL VOC "D:/Datasets/PASCAL_VOC/train.csv", "D:/Datasets/PASCAL_VOC/images", "D:/Datasets/PASCAL_VOC/labels",
    # MS COCO    "COCO/train.csv", "COCO/images/images/", "COCO/labels/labels_new/" 
    dataset = YOLODataset(
        csv_file=config.DATASET + "train.csv", # csv_file 
        image_dir=config.IMAGE_DIR,    # img_dir 
        label_dir=config.LABEL_DIR,    # label_dir 
        S=S, # S=[13, 26, 52],   
        anchors=anchors, 
        transform=transform, 
    )
    scaled_anchors = torch.tensor(anchors) / (1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    counter = 0 # count number of tests
    for x, y in loader:
        # print(f"x[0] shape: {x[0].shape}") # NOTE torch.Size([416, 416, 3])
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            # print(f"anchor.shape: {anchor.shape}") # torch.Size([3, 2])
            # print(f"y[{i}].shape: {y[i].shape}")
            # y[0].shape: torch.Size([1, 3, 13, 13, 6])
            # y[1].shape: torch.Size([1, 3, 26, 26, 6])
            # y[2].shape: torch.Size([1, 3, 52, 52, 6])
            boxes += cells_to_bboxes(y[i], is_preds=False, S=y[i].shape[2], anchors=anchor)[0]

        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        # print(f"boxes: {boxes}")
        # plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes) # torch.Size([416, 416, 3])

        # 
        # TypeError: Invalid shape (3, 416, 416) for image data

        # x[0].permute(0, 1, 2).shape is the original shape with torch.Size([3, 416, 416]) and it's an Invalid shape for image data
        # so x[0].permute(1, 2, 0).shape is the valid shape with torch.Size([416, 416, 3]) 
        # print("original shape: ", x[0].permute(0, 1, 2).shape) # torch.Size([3, 416, 416])
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes) # 
        print("-----------------------------------------")

        counter += 1 
        if counter == 1: break # run the test for some times then we stop

        # sometimes would run into out of bound ValueError, NOTE probabily caused by transforms, scale, and bbox_params settings!
        # File "C:\Users\paulc\.conda\envs\pt3.7\lib\site-packages\albumentations\augmentations\bbox_utils.py", line 330, in check_bbox
        #     "to be in the range [0.0, 1.0], got {value}.".format(bbox=bbox, name=name, value=value)
        # ValueError: Expected x_max for bbox (0.9375, 0.875, 1.0625, 1.0, 0.0) to be in the range [0.0, 1.0], got 1.0625.

        # NOTE this ValueError: Expected x_min for bbox (-0.01171875, 0.4296875, 0.09765625, 0.4609375, 2.0) to be in the range [0.0, 1.0], got -0.01171875.
        # is still unsolvable! I don't know why this error occur


if __name__ == "__main__":
    test()


