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
BATCH_SIZE = 16  # 32
IMAGE_SIZE = 416 # has to be multiple of 32
NUM_CLASSES = 3  # 20, 80
LEARNING_RATE = 15e-5  # 1e-4, 3e-4, 3e-5, 14e-5, 15e-5

WEIGHT_DECAY = 1e-4 # 1e-4
NUM_EPOCHS = 100    # 300
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

# Original YOLO on Pascal-VOC 
# ANCHORS = [
#     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], 
#     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
#     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
# ]  # Note these have been rescaled to be between [0, 1]

# YOLO-CFAR
# ANCHORS = [
#     [(0.1250, 0.1250), (0.1250, 0.1250), (0.1250, 0.1250)],
#     [(0.1250, 0.1250), (0.1250, 0.1250), (0.1250, 0.1250)],
#     [(0.1250, 0.1250), (0.1250, 0.1250), (0.1250, 0.1250)],
# ]

# sklearn.cluster.KMeans() 
# ANCHORS = [
#     [(0.21, 0.10), (0.34, 0.09), (0.50, 0.09)],
#     [(0.16, 0.03), (0.23, 0.04), (0.12, 0.08)], 
#     [(0.03, 0.02), (0.06, 0.03), (0.11, 0.02)], 
# ]

# sklearn.cluster.MiniBatchKMeans() 
# ANCHORS = [
#     [(0.211, 0.098), (0.339, 0.087), (0.495, 0.092)], 
#     [(0.158, 0.033), (0.232, 0.043), (0.125, 0.082)],
#     [(0.033, 0.017), (0.065, 0.027), (0.107, 0.024)],
# ]

### NOTE 6-fold 
# 0607
# ANCHORS = [
#     [(0.125, 0.073), (0.219, 0.097), (0.424, 0.095)],
#     [(0.040, 0.048), (0.121, 0.025), (0.219, 0.041)],
#     [(0.016, 0.016), (0.039, 0.009), (0.058, 0.019)],
# ]

# 0608
# ANCHORS = [
#     [(0.274, 0.045), (0.223, 0.095), (0.423, 0.096)],
#     [(0.096, 0.021), (0.158, 0.030), (0.128, 0.071)],
#     [(0.031, 0.008), (0.044, 0.019), (0.041, 0.049)],
# ]

# 0609 
# ANCHORS = [
#     [(0.245, 0.041), (0.228, 0.097), (0.423, 0.095)],
#     [(0.054, 0.046), (0.130, 0.027), (0.138, 0.072)], 
#     [(0.033, 0.009), (0.023, 0.028), (0.072, 0.017)], 
# ]

# 0610
# ANCHORS = [
#     [(0.269, 0.045), (0.211, 0.094), (0.418, 0.097)],
#     [(0.097, 0.021), (0.159, 0.029), (0.115, 0.065)],
#     [(0.031, 0.008), (0.044, 0.019), (0.034, 0.053)],
# ]

### NOTE 10-fold 
# new-anchors-0613-split0.txt 
# ANCHORS = [
#     [(0.2125, 0.0944), (0.3933, 0.0778), (0.4409, 0.1112)], 
#     [(0.1353, 0.0312), (0.1141, 0.0718), (0.2367, 0.0410)],
#     [(0.0346, 0.0108), (0.0397, 0.0346), (0.0971, 0.0188)], 
# ]

# new-anchors-0613-split1.txt 
# ANCHORS = [
#     [(0.2625, 0.0436), (0.2122, 0.0938), (0.4177, 0.0968)],
#     [(0.1127, 0.0200), (0.1411, 0.0324), (0.1123, 0.0710)], 
#     [(0.0288, 0.0097), (0.0570, 0.0164), (0.0362, 0.0389)],
# ]

# new-anchors-0614-split2.txt
# ANCHORS = [
#     [(0.2541, 0.0424), (0.2245, 0.0959), (0.4217, 0.0955)],
#     [(0.0558, 0.0446), (0.1402, 0.0278), (0.1344, 0.0724)],
#     [(0.0332, 0.0088), (0.0233, 0.0280), (0.0776, 0.0179)],
# ]

# new-anchors-0615-split3.txt
# ANCHORS = [
#     [(0.1995, 0.0927), (0.3784, 0.0672), (0.4196, 0.1032)],
#     [(0.1506, 0.0245), (0.1097, 0.0510), (0.2302, 0.0416)],
#     [(0.0336, 0.0115), (0.0367, 0.0437), (0.0814, 0.0198)],
# ]

# new-anchors-0615-split4.txt
# ANCHORS = [
#     [(0.2937, 0.0492), (0.2119, 0.0943), (0.4180, 0.0988)],
#     [(0.1057, 0.0222), (0.1801, 0.0301), (0.1170, 0.0637)],
#     [(0.0288, 0.0096), (0.0548, 0.0176), (0.0339, 0.0416)],
# ]

# new-anchors-0616-split5.txt
# ANCHORS = [
#     [(0.2272, 0.0397), (0.2119, 0.0932), (0.4167, 0.0934)],
#     [(0.0313, 0.0408), (0.1258, 0.0252), (0.1103, 0.0658)],
#     [(0.0269, 0.0099), (0.0655, 0.0103), (0.0623, 0.0202)],
# ]

# new-anchors-0616-split6.txt
# ANCHORS = [
#     [(0.1484, 0.0798), (0.2919, 0.0535), (0.3513, 0.1010)],
#     [(0.0480, 0.0241), (0.1096, 0.0268), (0.1907, 0.0312)],
#     [(0.0294, 0.0088), (0.0221, 0.0415), (0.0747, 0.0131)],
# ]

# new-anchors-0617-split7.txt
# ANCHORS = [
#     [(0.2748, 0.0458), (0.2125, 0.0939), (0.4158, 0.0972)],
#     [(0.1344, 0.0219), (0.1423, 0.0354), (0.1092, 0.0715)],
#     [(0.0313, 0.0103), (0.0729, 0.0177), (0.0369, 0.0373)],
# ]

# new-anchors-0617-split8.txt
# ANCHORS = [
#     [(0.2579, 0.0416), (0.1804, 0.0873), (0.4005, 0.0958)],
#     [(0.0326, 0.0506), (0.1152, 0.0179), (0.1304, 0.0344)],
#     [(0.0279, 0.0053), (0.0341, 0.0118), (0.0559, 0.0220)],
# ]

# new-anchors-0617-split9.txt
# ANCHORS = [
#     [(0.2558, 0.0430), (0.2129, 0.0937), (0.4190, 0.0953)],
#     [(0.1174, 0.0180), (0.1350, 0.0320), (0.1095, 0.0708)],
#     [(0.0325, 0.0095), (0.0301, 0.0367), (0.0599, 0.0204)],
# ]

### NOTE 4-fold
# new-anchors-0618-split0.txt
# ANCHORS = [
#     [(0.3048, 0.0498), (0.2005, 0.0912), (0.4144, 0.0989)],
#     [(0.0662, 0.0681), (0.1940, 0.0266), (0.1419, 0.0432)],
#     [(0.0344, 0.0096), (0.0448, 0.0242), (0.1047, 0.0211)],
# ]

# new-anchors-0618-split1.txt
# ANCHORS = [
#     [(0.2945, 0.0501), (0.2094, 0.0941), (0.4186, 0.0980)],
#     [(0.1126, 0.0241), (0.1824, 0.0326), (0.1081, 0.0664)],
#     [(0.0288, 0.0096), (0.0647, 0.0158), (0.0365, 0.0359)],
# ]

# new-anchors-0618-split2.txt
# ANCHORS = [
#     [(0.2355, 0.0427), (0.1963, 0.0911), (0.4061, 0.0955)],
#     [(0.0807, 0.0157), (0.1305, 0.0266), (0.0822, 0.0624)],
#     [(0.0344, 0.0086), (0.0156, 0.0268), (0.0448, 0.0277)],
# ]

# new-anchors-0619-split3.txt
# ANCHORS = [
#     [(0.2094, 0.0965), (0.3335, 0.0899), (0.4850, 0.0973)],
#     [(0.1330, 0.0321), (0.1271, 0.0730), (0.2507, 0.0406)],
#     [(0.0341, 0.0116), (0.0391, 0.0404), (0.0940, 0.0190)],
# ]

### NOTE 5-fold
# new-anchors-0619-split0.txt
# ANCHORS = [
#     [(0.2789, 0.0481), (0.2123, 0.0947), (0.4213, 0.0964)],
#     [(0.1068, 0.0223), (0.1779, 0.0296), (0.1180, 0.0636)],
#     [(0.0302, 0.0096), (0.0541, 0.0190), (0.0317, 0.0459)],
# ]

# new-anchors-0626-split1.txt
# ANCHORS = [
#     [(0.2578, 0.0419), (0.2227, 0.0962), (0.4202, 0.0954)],
#     [(0.0433, 0.0497), (0.1380, 0.0325), (0.1323, 0.0742)],
#     [(0.0314, 0.0083), (0.0447, 0.0194), (0.1097, 0.0193)],
# ]

# new-anchors-0626-split2.txt
# ANCHORS = [
#     [(0.2546, 0.0421), (0.2250, 0.0959), (0.4204, 0.0956)],
#     [(0.0473, 0.0484), (0.1361, 0.0292), (0.1342, 0.0723)],
#     [(0.0324, 0.0079), (0.0354, 0.0196), (0.0903, 0.0180)],
# ]

# new-anchors-0627-split3.txt
# ANCHORS = [
#     [(0.2803, 0.0469), (0.2126, 0.0939), (0.4151, 0.0968)],
#     [(0.0989, 0.0241), (0.1710, 0.0297), (0.1126, 0.0672)],
#     [(0.0281, 0.0097), (0.0633, 0.0142), (0.0349, 0.0356)],
# ]

# new-anchors-0627-split4.txt
# ANCHORS = [
#     [(0.2095, 0.0357), (0.2143, 0.0911), (0.4168, 0.0929)],
#     [(0.0343, 0.0555), (0.1138, 0.0227), (0.1172, 0.0627)],
#     [(0.0181, 0.0156), (0.0393, 0.0083), (0.0533, 0.0210)],
# ]

### NOTE 7-fold
# new-anchors-0628-split0.txt
# ANCHORS = [
#     [(0.2720, 0.0454), (0.1973, 0.0903), (0.4119, 0.0969)],
#     [(0.1135, 0.0198), (0.0764, 0.0589), (0.1516, 0.0332)],
#     [(0.0331, 0.0079), (0.0224, 0.0306), (0.0520, 0.0188)],
# ]

# new-anchors-0628-split1.txt
# ANCHORS = [
#     [(0.2746, 0.0466), (0.2114, 0.0945), (0.4157, 0.0965)],
#     [(0.0999, 0.0228), (0.1731, 0.0290), (0.1180, 0.0645)],
#     [(0.0286, 0.0093), (0.0527, 0.0157), (0.0349, 0.0399)],
# ]

# new-anchors-0628-split2.txt
# ANCHORS = [
#     [(0.2437, 0.0415), (0.2137, 0.0936), (0.4206, 0.0960)],
#     [(0.0346, 0.0394), (0.1310, 0.0266), (0.1123, 0.0675)],
#     [(0.0346, 0.0039), (0.0309, 0.0112), (0.0717, 0.0180)],
# ]

# new-anchors-0628-split3.txt
# ANCHORS = [
#     [(0.2744, 0.0462), (0.2191, 0.0960), (0.4186, 0.0962)],
#     [(0.0988, 0.0234), (0.1693, 0.0293), (0.1249, 0.0700)],
#     [(0.0322, 0.0083), (0.0472, 0.0180), (0.0322, 0.0460)],
# ]

# new-anchors-0628-split4.txt
ANCHORS = [
    [(0.2511, 0.0440), (0.2170, 0.0939), (0.4211, 0.0962)],
    [(0.0976, 0.0286), (0.1650, 0.0256), (0.1167, 0.0710)],
    [(0.0304, 0.0104), (0.0715, 0.0162), (0.0326, 0.0388)],
]

# 
# ANCHORS = [
#     [],
#     [],
#     [],
# ]

# 
# ANCHORS = [
#     [],
#     [],
#     [],
# ]

# 
# ANCHORS = [
#     [],
#     [],
#     [],
# ]


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
    


