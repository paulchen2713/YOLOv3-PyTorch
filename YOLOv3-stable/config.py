# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:01:48 2022

@patch: 
    2022.08.01
    2023.02.17
@author: Paul
@file: config.py
@dependencies:
    env pt3.7
    python 3.7.13
    numpy==1.19.2
    pytorch==1.7.1
    torchaudio==0.7.2
    torchvision==0.8.2
    albumentations==0.5.2
"""

import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
# from utils import seed_everything

DATASET = 'D:/Datasets/RD_maps/' # 'D:/Datasets/PASCAL_VOC', 'D:/Datasets/RD_maps'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# seed_everything()  # If you want deterministic behavior

NUM_WORKERS = 1      # 4
BATCH_SIZE = 10      # 32
IMAGE_SIZE = 416     # 416
NUM_CLASSES = 1     # PASCAL VOV has 20 classes, MS COCO has 80 classes
LEARNING_RATE = 1e-4 # 3e-5

WEIGHT_DECAY = 1e-4     # 1e-4, 5e-4
NUM_EPOCHS = 1      # 1000
CONF_THRESHOLD = 0.4 # 0.6
MAP_IOU_THRESH = 0.5 
NMS_IOU_THRESH = 0.45 

stride = [32, 16, 8] 
S = [IMAGE_SIZE // stride[0], IMAGE_SIZE // stride[1], IMAGE_SIZE // stride[2]] # [13, 26, 52]
# S = [IMAGE_SIZE // 8, IMAGE_SIZE // 4, IMAGE_SIZE // 2] # [2, 4, 8]

PIN_MEMORY = False # True
LOAD_MODEL = False # True
SAVE_MODEL = False # True

# "D:/Datasets/PASCAL_VOC/checkpoint.pth.tar", "D:/Datasets/RD_maps/checkpoint.pth.tar"
CHECKPOINT_FILE = DATASET + "checkpoint.pth.tar" # 
IMAGE_DIR = DATASET + "scaled_colors/"   # "/images/"
LABEL_DIR = DATASET + "labels/" # "/labels/"

# how we handle the anchor boxes? we will specify the anchor boxes in the following manner as a list of lists 
# of tuples, where each tuple corresponds to the width and the height of a anchor box relative to the image size 
# and each list grouping together three tuples correspond to the anchors used on a specific prediction scale

# ANCHORS = [
#     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # largest anchor boxes
#     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], # medium anchor boxes
#     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], # small anchor boxes
# ]  
# note these anchors above are the ones used in the original paper and have been rescaled to be between [0, 1]

ANCHORS = [
    [(0.1250, 0.1250), (0.1250, 0.1250), (0.1250, 0.1250)],
    [(0.1250, 0.1250), (0.1250, 0.1250), (0.1250, 0.1250)],
    [(0.1250, 0.1250), (0.1250, 0.1250), (0.1250, 0.1250)],
]

# scale cannot be lower than 1.0, otherwise would encounter ValueError: Requested crop size (416, 416) is larger than the image size (332, 332)
# scale = 1.0 # 1.1, 1.2 
train_transforms = A.Compose(
    [
        # A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        # A.PadIfNeeded(
        #     min_height=int(IMAGE_SIZE * scale), min_width=int(IMAGE_SIZE * scale), 
        #     # border_mode=cv2.BORDER_CONSTANT,
        # ),
        # A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        # A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        # A.OneOf(
        #     [
        #         A.ShiftScaleRotate(
        #             rotate_limit=10, p=0.4, border_mode=cv2.BORDER_CONSTANT
        #         ),
        #         A.IAAAffine(shear=10, p=0.4, mode="constant"),
        #     ],
        #     p=1.0,
        # ),
        # A.HorizontalFlip(p=0.5),
        # A.Blur(p=0.1),
        # A.CLAHE(p=0.1),
        # A.Posterize(p=0.1),
        # A.ToGray(p=0.1),
        # A.ChannelShuffle(p=0.05),

        # remove Normalize may cause RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.HalfTensor) should be the same
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    # bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],), # transform on bbox_params would cause unknown error
)
test_transforms = A.Compose(
    [
        # A.LongestMaxSize(max_size=IMAGE_SIZE),
        # A.PadIfNeeded(
        #     min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, 
        #     # border_mode=cv2.BORDER_CONSTANT
        # ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    # bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]), # transform on bbox_params would cause unknown error
)

# ValueError: Expected y_max for bbox (0.375, 0.9375, 0.5, 1.0625, 0.0) to be in the range [0.0, 1.0], got 1.0625.
# train_transforms, test_transforms = None, None # transform would cause unknown error

CLASSES = [
    "target"
]

# PASCAL_CLASSES, remember to rename it back to "CLASSES" when using Pascal VOC Dataset
CLASSES2 = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

# COCO_LABELS, remember to rename it back to "CLASSES" when using COCO Dataset
CLASSES3 = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]

# Albumentations examples (https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_bboxes.ipynb)
def test():
    # Import the required libraries, besides albumentations and cv2
    import random
    # from PIL import Image
    import numpy as np
    from matplotlib import pyplot as plt
    
    # Define functions to visualize bounding boxes and class labels on an image
    BOX_COLOR = (255, 0, 0)      # Red
    TEXT_COLOR = (255, 255, 255) # White

    def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
        """Visualizes a single bounding box on the image"""
        # YOLO format
        x, y, w, h = bbox
        x_min, x_max = int((2*x - h) / 2), int((2*x + h) / 2)
        y_min, y_max = int((2*y - w) / 2), int((2*y + w) / 2)

        # COCO format
        # x_min, y_min, w, h = bbox
        # x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35, 
            color=TEXT_COLOR, 
            lineType=cv2.LINE_AA,
        )
        return img

    def visualize(image, bboxes, category_ids, category_id_to_name):
        img = image.copy()
        for bbox, category_id in zip(bboxes, category_ids):
            class_name = category_id_to_name[category_id]
            img = visualize_bbox(img, bbox, class_name)
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()

    # Load the image and the annotations for it
    img_idx = random.randint(1, 1000) # get a random image index
    print(f"image: {img_idx}.txt") 
    # we can read the image through cv2.imread() in BGR or PIL.Image.open() in RGB, but the visualiz() 
    # and visualize_bbox() functions are implemented with cv2, so we should stick to it to avoid errors
    img_path = IMAGE_DIR + f'{img_idx}_sc.jpg'
    image = cv2.imread(img_path) # NOTE cv2.imread() read the image in BGR, 0~255, (W, H, C)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # must first convert BGR into RGB

    label_path = LABEL_DIR + f'{img_idx}.txt'
    label = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()
    # print(label[0][1:])
    true_scale = [label[0][i]*IMAGE_SIZE for i in range(1, 5)]
    # print(true_scale)

    bboxes = list()
    bboxes.append(true_scale)
    print(f"bbox: {bboxes}")
    
    # bboxes, category_ids and category_id_to_name all has to be iterable object
    category_ids = [0]
    category_id_to_name = {0: 'target'}

    # Visuaize the original image with bounding boxes
    # visualize(image, bboxes, category_ids, category_id_to_name)

    # Define an augmentation pipeline
    tscale = 1.5
    transform = A.Compose(
        [
            # A.LongestMaxSize(max_size=int(IMAGE_SIZE * tscale), p=1.0), 
            # A.PadIfNeeded(min_height=int(IMAGE_SIZE * tscale), min_width=int(IMAGE_SIZE * tscale), border_mode=cv2.BORDER_CONSTANT, ),

            # A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE, p=1.0), 

            # A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=1.0), 
            
            # A.ShiftScaleRotate(rotate_limit=20, p=1.0, border_mode=cv2.BORDER_CONSTANT), 
            # A.HorizontalFlip(p=1.0), 
            # A.Blur(blur_limit=7, p=1.0), 
            # A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0), 
            # A.Posterize(p=1.0),
            # A.ToGray(p=1.0),             
            # A.ChannelShuffle(p=1.0), 
            # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0),
        ],
        bbox_params=A.BboxParams(
            format='coco', 
            label_fields=['category_ids']
        ),
    )
    random.seed(33)
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name)
    # Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)


if __name__ == "__main__":
    test()

