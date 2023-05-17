# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:10:36 2023

@patch: 
    2023.03.13
    2023.03.22

@author: Paul
@file: config.py
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

import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
# from utils import seed_everything
import numpy as np
import os
import random

PATH = "D:/Datasets/YOLOv3-PyTorch/"
DATASET = 'D:/Datasets/RADA/RD_JPG/' # RD_416
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

NUM_WORKERS = 1  # num of threads
BATCH_SIZE = 20  # 32
IMAGE_SIZE = 416 # has to be multiple of 32
NUM_CLASSES = 3  # 20, 80
LEARNING_RATE = 28e-5  # 3e-5 1e-4

WEIGHT_DECAY = 1e-4 # 1e-4
NUM_EPOCHS = 80    # 300
CONF_THRESHOLD = 0.6  # 0.6
MAP_IOU_THRESH = 0.5  # 0.5
NMS_IOU_THRESH = 0.45 # 0.45

TEST_POINT = 10 # compute the test accuracies and mAP for every TEST_POINT number of epochs


stride = [32, 16, 8] 
S = [IMAGE_SIZE // stride[0], IMAGE_SIZE // stride[1], IMAGE_SIZE // stride[2]] # [13, 26, 52]

PIN_MEMORY = False # True
LOAD_MODEL = False # True
SAVE_MODEL = True # True

# "checkpoint.pth.tar" "YOLOv3-pretrained-weights/pytorch_format/yolov3_pascal_78.1map.pth.tar"
CHECKPOINT_FILE = PATH + "yolov3_pascal_voc.pth.tar" 

IMAGE_DIR = DATASET + "imagesc/"
LABEL_DIR = DATASET + "labels/"

# how we handle the anchor boxes? we will specify the anchor boxes in the following manner as a list of lists 
# of tuples, where each tuple corresponds to the width and the height of a anchor box relative to the image size 
# and each list grouping together three tuples correspond to the anchors used on a specific prediction scale

# ANCHORS = [
#     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
#     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
#     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
# ]  # Note these have been rescaled to be between [0, 1]

# ANCHORS = [
#     [(0.1250, 0.1250), (0.1250, 0.1250), (0.1250, 0.1250)],
#     [(0.1250, 0.1250), (0.1250, 0.1250), (0.1250, 0.1250)],
#     [(0.1250, 0.1250), (0.1250, 0.1250), (0.1250, 0.1250)],
# ]

# # result of sklearn.cluster.KMeans() 
# ANCHORS = [
#     [(0.21, 0.10), (0.34, 0.09), (0.50, 0.09)],
#     [(0.16, 0.03), (0.23, 0.04), (0.12, 0.08)], 
#     [(0.03, 0.02), (0.06, 0.03), (0.11, 0.02)], 
# ]

# ANCHORS = [
#     [(0.211, 0.098), (0.339, 0.087), (0.495, 0.092)], 
#     [(0.158, 0.033), (0.232, 0.043), (0.125, 0.082)],
#     [(0.033, 0.017), (0.065, 0.027), (0.107, 0.024)],
# ]

ANCHORS = [
    [(0.125, 0.073), (0.219, 0.097), (0.424, 0.095)],
    [(0.040, 0.048), (0.121, 0.025), (0.219, 0.041)],
    [(0.016, 0.016), (0.039, 0.009), (0.058, 0.019)],
]



scale = 1.0 # 1.1
train_transforms = A.Compose(
    [
        # NOTE we need LongestMaxSize() and PadIfNeeded() to avoid RuntimeError: Trying to resize storage that is not resizable
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        # A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        # A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        # A.OneOf(
        #     [
        #         A.ShiftScaleRotate(
        #             rotate_limit=10, p=0.4, border_mode=cv2.BORDER_CONSTANT
        #         ),
        #         # A.IAAAffine(shear=10, p=0.4, mode="constant"),
        #     ],
        #     p=1.0,
        # ),
        # A.HorizontalFlip(p=0.5),
        # A.Blur(p=0.1),
        # A.CLAHE(p=0.1),
        # A.Posterize(p=0.1),
        # A.ToGray(p=0.1),
        # A.ChannelShuffle(p=0.05),
        # NOTE we need Normalize() to avoid RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.HalfTensor) should be the same
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,), 
        # NOTE we need ToTensorV2() to avoid RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[10, 416, 416, 3] to have 3 channels, but got 416 channels instead
        ToTensorV2(), 
    ],
    bbox_params=A.BboxParams(
        format="yolo", 
        # min_visibility=0.4, 
        # label_fields=[],
    ),
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(
        format="yolo", 
        # min_visibility=0.4, 
        # label_fields=[]
    ),
)


# CFAR_CLASS
CLASSES0 = [
    "target"
]

# CARRADA_CLASSES
CLASSES = [
    'person',
    'cyclist',
    'car',
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
    img_idx = random.randint(1, 999) # get a random image index # '000001' # 
    print(f"image: {img_idx}.txt") 
    # we can read the image through cv2.imread() in BGR or PIL.Image.open() in RGB, but the visualiz() 
    # and visualize_bbox() functions are implemented with cv2, so we should stick to it to avoid errors
    img_path = IMAGE_DIR + f'{img_idx}.jpg'
    image = cv2.imread(img_path) # NOTE cv2.imread() read the image in BGR, 0~255, (W, H, C)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # must first convert BGR into RGB

    label_path = LABEL_DIR + f'{img_idx}.txt'
    label = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()
    true_scale = [label[0][i]*IMAGE_SIZE for i in range(1, 5)]
    print("label: ", label[0][1:])
    print("true scale: ", true_scale)

    bboxes = list()
    bboxes.append(true_scale)
    print(f"bbox: {bboxes}")
    
    # bboxes, category_ids and category_id_to_name all has to be iterable object
    # RD_maps dataset
    # rdmap_ids = [0]
    # rdmap_id_to_name = {0: 'target'}

    # PASCAL_VOC dataset
    # category_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] # pascal_voc_ids = [i for i in range(0, 20)]
    # category_id_to_name = {
    #     0: "aeroplane",
    #     1: "bicycle",
    #     2: "bird",
    #     3: "boat",
    #     4: "bottle",
    #     5: "bus",
    #     6: "car",
    #     7: "cat",
    #     8: "chair",
    #     9: "cow",
    #     10: "diningtable",
    #     11: "dog",
    #     12: "horse",
    #     13: "motorbike",
    #     14: "person",
    #     15: "pottedplant",
    #     16: "sheep",
    #     17: "sofa",
    #     18: "train",
    #     19: "tvmonitor"
    # }

    # CARRADA dataset
    category_ids = [0, 1, 2]
    category_id_to_name = {0: 'person', 1: 'cyclist', 2: 'car'}

    # Visuaize the original image with bounding boxes
    # visualize(image, bboxes, category_ids, category_id_to_name)

    # Define an augmentation pipeline
    tscale = 1.0
    transform = A.Compose(
        [
            A.LongestMaxSize(max_size=int(IMAGE_SIZE * tscale), p=1.0), 
            A.PadIfNeeded(min_height=int(IMAGE_SIZE * tscale), min_width=int(IMAGE_SIZE * tscale), border_mode=cv2.BORDER_CONSTANT, ),

            # A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE, p=1.0), 

            # A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=1.0), 
            
            # A.ShiftScaleRotate(rotate_limit=20, p=1.0, border_mode=cv2.BORDER_CONSTANT), 
            # A.HorizontalFlip(p=1.0), 
            # A.Blur(blur_limit=7, p=1.0), 
            # A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0), 
            # A.Posterize(p=1.0),
            # A.ToGray(p=1.0),             
            # A.ChannelShuffle(p=1.0), 
            
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format='coco', # 'coco', 'yolo'
            label_fields=['category_ids'] # 'category_ids', 'pascal_voc_ids'
        ),
    )
    # random.seed(33)
    print(bboxes)
    print(label)
    transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
    # visualize(
    #     image=transformed['image'], 
    #     bboxes=transformed['bboxes'], 
    #     category_ids=transformed['category_ids'], 
    #     category_id_to_name=category_id_to_name
    # )
    visualize(
        image=image, 
        bboxes=bboxes,
        category_ids=category_ids,
        category_id_to_name=category_id_to_name
    )
    # Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)


if __name__ == "__main__":
    print("test() from config.py has been disabled")
    # test()   
    


