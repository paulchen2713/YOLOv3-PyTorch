# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:09:16 2023

@patch: 
    2023.02.17
    2023.03.22

@author: Paul
@file: model.py
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
    torchinfo==1.7.2

@references:
    Redmon, Joseph and Farhadi, Ali, YOLOv3: An Incremental Improvement, April 8, 2018. (https://doi.org/10.48550/arXiv.1804.02767)
    Ayoosh Kathuria, Whats new in YOLO v3?, April, 23, 2018. (https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)
    Sanna Persson, YOLOv3 from Scratch, Mar 21, 2021. (https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0)

Implementation of YOLOv3 architecture
"""


import torch
import torch.nn as nn
from torchinfo import summary
import torchsummary

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""

config = [
    (32, 3, 1),   # (32, 3, 1) is the CBL, CBL = Conv + BN + LeakyReLU
    (64, 3, 2),
    ["B", 1],     # (64, 3, 2) + ["B", 1] is the Res1, Res1 = ZeroPadding + CBL + (CBL + CBL + Add)*1
    (128, 3, 2),
    ["B", 2],     # (128, 3, 2) + ["B", 2] is th Res2, Res2 = ZeroPadding + CBL + (CBL + CBL + Add)*2
    (256, 3, 2),
    ["B", 8],     # (256, 3, 2) + ["B", 8] is th Res8, Res8 = ZeroPadding + CBL + (CBL + CBL + Add)*8
    (512, 3, 2),
    ["B", 8],     # (512, 3, 2) + ["B", 8] is th Res8, Res8 = ZeroPadding + CBL + (CBL + CBL + Add)*8
    (1024, 3, 2),
    ["B", 4],     # (1024, 3, 2) + ["B", 4] is th Res4, Res4 = ZeroPadding + CBL + (CBL + CBL + Add)*4
    # to this point is Darknet-53 which has 52 layers
    # 52 = 1 + (1 + 1*2) + (1 + 2*2) + (1 + 8*2) + (1 + 8*2) + (1 + 4*2) ?
    (512, 1, 1),  # 
    (1024, 3, 1), #
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
    # 252 = 1 + 3 + (4+7) + (4+7*2) + (4+7*8) + (4+7*8) + (4+7*4) + 19 + 5 + 19 + 5 + 19 ?
]

config = [
    (16, 3, 1),   
    (32, 3, 2),
    ["B", 1],
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),   ## 1 
    ["B", 2],     
    (256, 3, 2),  ## 2 
    ["B", 2],     
    (512, 3, 2),  ## 3 
    ["B", 1],     
    (256, 1, 1),  
    (512, 3, 1),  ## 3 
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),  ## 2 
    "S",
    (64, 1, 1),
    "U",
    (64, 1, 1),
    (128, 3, 1),   ## 1 
    "S",
]

config = [
    (8, 3, 1),   
    (16, 3, 2),
    ["B", 1],
    (32, 3, 2),
    ["B", 1],
    (64, 3, 2),   ## 1 
    ["B", 2],     
    (128, 3, 2),  ## 2 
    ["B", 2],     
    (256, 3, 2),  ## 3 
    ["B", 1],     
    (128, 1, 1),  
    (256, 3, 1),  ## 3 
    "S",
    (64, 1, 1),
    "U",
    (64, 1, 1),
    (128, 3, 1),  ## 2 
    "S",
    (32, 1, 1),
    "U",
    (32, 1, 1),
    (64, 3, 1),   ## 1 
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(CNNBlock, self).__init__()
        # if we do use bn activation function in the block, then we do not want to use bias, its unnecessary
        # **kwargs will be the kernal size, the stride and padding as well
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs) 
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(negative_slope=0.1) # default negative_slope=0.01
        self.use_bn_act = bn_act # indicating if the block is going to use a batch norm NN activation function

    def forward(self, x):
        # using if-else statement in the forward pass might lose on some performance, negligible?
        # we use bn activation by default, except for scale prediction
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x))) # bn_act()
        # for scale prediction we don't want to use batch norm LeakyReLU on our output, just normal Conv
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats): # repeat for num_repeats
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1, padding=0), # down samples or reduces the number of filters
                    # CNNBlock(channels // 2, channels, kernel_size=3, padding=1), # then brings it back again
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1), 
                )
            ]
        # 1. why specify use_residual in a ResidualBlock? is because in some cases we are going to use skip 
        # connections, in some cases we just going through the config file and build the ordinary ResidualBlock
        # 2. why we need to store these? 
        self.use_residual = use_residual # indicating using residual
        self.num_repeats = num_repeats   # number of repeats set to 1 by default

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
            # if self.use_residual:
            #     # x = x + layer(x)
            #     x = layer(x) + x
            # else:
            #     x = layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ScalePrediction, self).__init__()
        # for every single cell grid we have 3 anchor boxes, for every anchor box we have 1 node for each of the classes
        # for each bounding box we have [P(Object), x, y, w, h] and that's 5 values
        self.pred = nn.Sequential(
            # CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1), 
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1), 
            CNNBlock(2 * in_channels, 3 * (num_classes + 5), bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        # we want to return the prediction of x, then we want to reshape it to the number of examples in our batch
        # split out_channel "3 * (num_classes + 5)" into two different dimensions "3, (num_classes + 5)", instead of 
        # having a long vector of bounging boxes, and change the order of the dimensions
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]) 
            .permute(0, 1, 3, 4, 2) 
        ) 
        # [x.shape[0], 3, x.shape[2], x.shape[3], self.num_classes + 5], e.g. [N, 3, 13, 13, 5+num_classes]
        # for scale one, we have N examples in our batch, each example has 3 anchors, each anchor has 13-by-13 grid
        # and every cell has (5+num_classes) output, output dimension = N x 3 x 13 x 13 x (5+num_classes)


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        # we want to create the layers using the config file, and store them in a nn.ModuleList()
        self.layers = self._create_conv_layers() # we immediately call _create_conv_layers() to initialize the layers

    def forward(self, x):
        # need to keep track of outputs and route connections
        outputs = []           # we have one output for each scale prediction, should be 3 in total
        route_connections = [] # e.g. after upsampling, we concatenate the channels of skip connections

        for i, layer in enumerate(self.layers):
            if isinstance(layer, ScalePrediction): # if it's ScalePrediction
                outputs.append(layer(x)) # we're going to add that layer
                continue # and then continue from where we were previously, not after ScalePrediction

            # calling layer(x) is equivalent to calling layers.__call__(x), and __call__() is actually calling layer.forward(x)
            # which is defined in class layer(nn.Module), but in practice we should use layer(x) rather than layer.forward(x)
            x = layer(x) # 
            # print(f"layer {i}: ", x.shape) # layer 0:  torch.Size([16, 32, 416, 416])

            # skip layers are connected to ["B", 8] based on the paper, original config file 
            # if isinstance(layer, ResidualBlock) and layer.num_repeats != 1: # 
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 2:  # NOTE 8
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample): # if we use the Upsample
                # we want to concatenates with the last route connection, with the last one we added
                x = torch.cat([x, route_connections[-1]], dim=1)  # why concatenate along dimension 1 for the channels
                route_connections.pop() # after concatenation, we remove the last one

        # print(f"outputs: {outputs}")
        return outputs

    # create the layers using the config files
    def _create_conv_layers(self):
        layers = nn.ModuleList()       # keep track of all the layers in a ModuleList, which supports tools like model.eval() 
        in_channels = self.in_channels # only need to specifies the first in_channels, I suppose

        # go through and parse the config file and construct the model line by line
        for module in config:
            # if it's a tuple (filters, kernel_size, stride), e.g. (32, 3, 1), then it's just a CNNBlock
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module # we want to take out the (filters, kernel_size, stride)
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        # padding=1 if kernel_size == 3 else 0, # if kernel_size == 1 then padding = 0
                        padding=1 if kernel_size == 3 else 0, 
                    )
                )
                # the in_channels for the next block is going to be the out_channels of this block
                in_channels = out_channels # update the in_channels of the next layer

            # if it's a List, e.g. ["B", 1], then it's a ResidualBlock
            elif isinstance(module, list):
                num_repeats = module[1] # we want to take out the number of repeats, which is going to be module[1]
                # and module[0] should be "B", which indicates that this is a ResidualBlock
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            # if it's a String, e.g. "S" or "U", then it might be ScalePrediction or Upsampling
            elif isinstance(module, str):
                # "S" for ScalePrediction
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    # after ScalePrediction, we want to continue from CNNBlock, since we have scale_factor=2
                    in_channels = in_channels // 2 # we then wnat to divide in_channels by 2
                # "U" for Upsampling
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3 # 3 == 2 + 1, concatenated the channels from previously
        
        return layers


if __name__ == "__main__":
    # actual parameters
    num_classes = 3 # 20
    
    # YOLOv1: 448, YOLOv2/YOLOv3: 416 (with multi-scale training)
    IMAGE_SIZE = 416 # multiples of 32 are workable with stride [32, 16, 8]
    # stride = [8, 4, 2] 
    # stride = [16, 8, 4] # 16
    stride = [32, 16, 8] # 32

    # simple test settings
    batch_size = 16  # num_samples
    num_channels = 3 # num_anchors
    
    model = YOLOv3(num_classes=num_classes) # initialize a YOLOv3 model as model
    
    # simple test with random inputs of 16 examples, 3 channels, and IMAGE_SIZE-by-IMAGE_SIZE input
    x = torch.randn((batch_size, num_channels, IMAGE_SIZE, IMAGE_SIZE))
    
    out = model(x) 

    # print out the model summary using third-party library called 'torchsummary'
    torchsummary.summary(model.cuda(), (num_channels, IMAGE_SIZE, IMAGE_SIZE), batch_size)

    # print out the model summary using torchinfo.summary()
    # summary(model.cuda(), input_size=(batch_size, num_channels, IMAGE_SIZE, IMAGE_SIZE))

    # print(model)

    print("Output Shape: ")
    print("[num_examples, num_channels, feature_map, feature_map, num_classes + 5]")
    for i in range(num_channels):
        print(out[i].shape)
    
    assert out[0].shape == (batch_size, num_channels, IMAGE_SIZE//stride[0], IMAGE_SIZE//stride[0], num_classes + 5)  # [20, 3, 13, 13, num_classes + 5]
    assert out[1].shape == (batch_size, num_channels, IMAGE_SIZE//stride[1], IMAGE_SIZE//stride[1], num_classes + 5)  # [20, 3, 26, 26, num_classes + 5]
    assert out[2].shape == (batch_size, num_channels, IMAGE_SIZE//stride[2], IMAGE_SIZE//stride[2], num_classes + 5)  # [20, 3, 52, 52, num_classes + 5]
    
    print("Success!")



# (pt3.8) C:\Users\Paul\Downloads\YOLOv3>D:/ProgramData/Anaconda3/envs/pt3.8/python.exe c:/Users/Paul/Downloads/YOLOv3/YOLOv3-CARRADA/model.py
# layer 0:  torch.Size([20, 32, 416, 416])
# layer 1:  torch.Size([20, 64, 208, 208])
# layer 2:  torch.Size([20, 64, 208, 208])
# layer 3:  torch.Size([20, 128, 104, 104])
# layer 4:  torch.Size([20, 128, 104, 104])
# layer 5:  torch.Size([20, 256, 52, 52])
# layer 6:  torch.Size([20, 256, 52, 52])
# layer 7:  torch.Size([20, 512, 26, 26])
# layer 8:  torch.Size([20, 512, 26, 26])
# layer 9:  torch.Size([20, 1024, 13, 13])
# layer 10:  torch.Size([20, 1024, 13, 13])
# layer 11:  torch.Size([20, 512, 13, 13])
# layer 12:  torch.Size([20, 1024, 13, 13])
# layer 13:  torch.Size([20, 1024, 13, 13])
# layer 14:  torch.Size([20, 512, 13, 13])
# layer 16:  torch.Size([20, 256, 13, 13])
# layer 17:  torch.Size([20, 256, 26, 26])
# layer 18:  torch.Size([20, 256, 26, 26])
# layer 19:  torch.Size([20, 512, 26, 26])
# layer 20:  torch.Size([20, 512, 26, 26])
# layer 21:  torch.Size([20, 256, 26, 26])
# layer 23:  torch.Size([20, 128, 26, 26])
# layer 24:  torch.Size([20, 128, 52, 52])
# layer 25:  torch.Size([20, 128, 52, 52])
# layer 26:  torch.Size([20, 256, 52, 52])
# layer 27:  torch.Size([20, 256, 52, 52])
# layer 28:  torch.Size([20, 128, 52, 52])

# Output Shape: 
# [num_examples, num_channels, feature_map, feature_map, num_classes + 5]
# torch.Size([20, 3, 13, 13, 8])
# torch.Size([20, 3, 26, 26, 8])
# torch.Size([20, 3, 52, 52, 8])
# Success!



# The result of torchinfo.summary()
# ====================================================================================================
# Layer (type:depth-idx)                             Output Shape              Param #
# ====================================================================================================
# YOLOv3                                             [20, 3, 13, 13, 8]        --
# ├─ModuleList: 1-1                                  --                        --
# │    └─CNNBlock: 2-1                               [20, 32, 416, 416]        --
# │    │    └─Conv2d: 3-1                            [20, 32, 416, 416]        864
# │    │    └─BatchNorm2d: 3-2                       [20, 32, 416, 416]        64
# │    │    └─LeakyReLU: 3-3                         [20, 32, 416, 416]        --
# │    └─CNNBlock: 2-2                               [20, 64, 208, 208]        --
# │    │    └─Conv2d: 3-4                            [20, 64, 208, 208]        18,432
# │    │    └─BatchNorm2d: 3-5                       [20, 64, 208, 208]        128
# │    │    └─LeakyReLU: 3-6                         [20, 64, 208, 208]        --
# │    └─ResidualBlock: 2-3                          [20, 64, 208, 208]        --
# │    │    └─ModuleList: 3-7                        --                        20,672
# │    └─CNNBlock: 2-4                               [20, 128, 104, 104]       --
# │    │    └─Conv2d: 3-8                            [20, 128, 104, 104]       73,728
# │    │    └─BatchNorm2d: 3-9                       [20, 128, 104, 104]       256
# │    │    └─LeakyReLU: 3-10                        [20, 128, 104, 104]       --
# │    └─ResidualBlock: 2-5                          [20, 128, 104, 104]       --
# │    │    └─ModuleList: 3-11                       --                        164,608
# │    └─CNNBlock: 2-6                               [20, 256, 52, 52]         --
# │    │    └─Conv2d: 3-12                           [20, 256, 52, 52]         294,912
# │    │    └─BatchNorm2d: 3-13                      [20, 256, 52, 52]         512
# │    │    └─LeakyReLU: 3-14                        [20, 256, 52, 52]         --
# │    └─ResidualBlock: 2-7                          [20, 256, 52, 52]         --
# │    │    └─ModuleList: 3-15                       --                        2,627,584
# │    └─CNNBlock: 2-8                               [20, 512, 26, 26]         --
# │    │    └─Conv2d: 3-16                           [20, 512, 26, 26]         1,179,648
# │    │    └─BatchNorm2d: 3-17                      [20, 512, 26, 26]         1,024
# │    │    └─LeakyReLU: 3-18                        [20, 512, 26, 26]         --
# │    └─ResidualBlock: 2-9                          [20, 512, 26, 26]         --
# │    │    └─ModuleList: 3-19                       --                        10,498,048
# │    └─CNNBlock: 2-10                              [20, 1024, 13, 13]        --
# │    │    └─Conv2d: 3-20                           [20, 1024, 13, 13]        4,718,592
# │    │    └─BatchNorm2d: 3-21                      [20, 1024, 13, 13]        2,048
# │    │    └─LeakyReLU: 3-22                        [20, 1024, 13, 13]        --
# │    └─ResidualBlock: 2-11                         [20, 1024, 13, 13]        --
# │    │    └─ModuleList: 3-23                       --                        20,983,808
# │    └─CNNBlock: 2-12                              [20, 512, 13, 13]         --
# │    │    └─Conv2d: 3-24                           [20, 512, 13, 13]         524,288
# │    │    └─BatchNorm2d: 3-25                      [20, 512, 13, 13]         1,024
# │    │    └─LeakyReLU: 3-26                        [20, 512, 13, 13]         --
# │    └─CNNBlock: 2-13                              [20, 1024, 13, 13]        --
# │    │    └─Conv2d: 3-27                           [20, 1024, 13, 13]        4,718,592
# │    │    └─BatchNorm2d: 3-28                      [20, 1024, 13, 13]        2,048
# │    │    └─LeakyReLU: 3-29                        [20, 1024, 13, 13]        --
# │    └─ResidualBlock: 2-14                         [20, 1024, 13, 13]        --
# │    │    └─ModuleList: 3-30                       --                        5,245,952
# │    └─CNNBlock: 2-15                              [20, 512, 13, 13]         --
# │    │    └─Conv2d: 3-31                           [20, 512, 13, 13]         524,288
# │    │    └─BatchNorm2d: 3-32                      [20, 512, 13, 13]         1,024
# │    │    └─LeakyReLU: 3-33                        [20, 512, 13, 13]         --
# │    └─ScalePrediction: 2-16                       [20, 3, 13, 13, 8]        --
# │    │    └─Sequential: 3-34                       [20, 24, 13, 13]          4,745,288
# │    └─CNNBlock: 2-17                              [20, 256, 13, 13]         --
# │    │    └─Conv2d: 3-35                           [20, 256, 13, 13]         131,072
# │    │    └─BatchNorm2d: 3-36                      [20, 256, 13, 13]         512
# │    │    └─LeakyReLU: 3-37                        [20, 256, 13, 13]         --
# │    └─Upsample: 2-18                              [20, 256, 26, 26]         --
# │    └─CNNBlock: 2-19                              [20, 256, 26, 26]         --
# │    │    └─Conv2d: 3-38                           [20, 256, 26, 26]         196,608
# │    │    └─BatchNorm2d: 3-39                      [20, 256, 26, 26]         512
# │    │    └─LeakyReLU: 3-40                        [20, 256, 26, 26]         --
# │    └─CNNBlock: 2-20                              [20, 512, 26, 26]         --
# │    │    └─Conv2d: 3-41                           [20, 512, 26, 26]         1,179,648
# │    │    └─BatchNorm2d: 3-42                      [20, 512, 26, 26]         1,024
# │    │    └─LeakyReLU: 3-43                        [20, 512, 26, 26]         --
# │    └─ResidualBlock: 2-21                         [20, 512, 26, 26]         --
# │    │    └─ModuleList: 3-44                       --                        1,312,256
# │    └─CNNBlock: 2-22                              [20, 256, 26, 26]         --
# │    │    └─Conv2d: 3-45                           [20, 256, 26, 26]         131,072
# │    │    └─BatchNorm2d: 3-46                      [20, 256, 26, 26]         512
# │    │    └─LeakyReLU: 3-47                        [20, 256, 26, 26]         --
# │    └─ScalePrediction: 2-23                       [20, 3, 26, 26, 8]        --
# │    │    └─Sequential: 3-48                       [20, 24, 26, 26]          1,193,032
# │    └─CNNBlock: 2-24                              [20, 128, 26, 26]         --
# │    │    └─Conv2d: 3-49                           [20, 128, 26, 26]         32,768
# │    │    └─BatchNorm2d: 3-50                      [20, 128, 26, 26]         256
# │    │    └─LeakyReLU: 3-51                        [20, 128, 26, 26]         --
# │    └─Upsample: 2-25                              [20, 128, 52, 52]         --
# │    └─CNNBlock: 2-26                              [20, 128, 52, 52]         --
# │    │    └─Conv2d: 3-52                           [20, 128, 52, 52]         49,152
# │    │    └─BatchNorm2d: 3-53                      [20, 128, 52, 52]         256
# │    │    └─LeakyReLU: 3-54                        [20, 128, 52, 52]         --
# │    └─CNNBlock: 2-27                              [20, 256, 52, 52]         --
# │    │    └─Conv2d: 3-55                           [20, 256, 52, 52]         294,912
# │    │    └─BatchNorm2d: 3-56                      [20, 256, 52, 52]         512
# │    │    └─LeakyReLU: 3-57                        [20, 256, 52, 52]         --
# │    └─ResidualBlock: 2-28                         [20, 256, 52, 52]         --
# │    │    └─ModuleList: 3-58                       --                        328,448
# │    └─CNNBlock: 2-29                              [20, 128, 52, 52]         --
# │    │    └─Conv2d: 3-59                           [20, 128, 52, 52]         32,768
# │    │    └─BatchNorm2d: 3-60                      [20, 128, 52, 52]         256
# │    │    └─LeakyReLU: 3-61                        [20, 128, 52, 52]         --
# │    └─ScalePrediction: 2-30                       [20, 3, 52, 52, 8]        --
# │    │    └─Sequential: 3-62                       [20, 24, 52, 52]          301,640
# ====================================================================================================
# Total params: 61,534,648
# Trainable params: 61,534,648
# Non-trainable params: 0
# Total mult-adds (G): 653.05
# ====================================================================================================
# Input size (MB): 41.53
# Forward/backward pass size (MB): 12265.99
# Params size (MB): 246.14
# Estimated Total Size (MB): 12553.66
# ====================================================================================================



# ----------------------------------------------------------------
# Total params: 61,534,504
# Trainable params: 61,534,504
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 39.61
# Forward/backward pass size (MB): 8276.82
# Params size (MB): 234.74
# Estimated Total Size (MB): 8551.17
# ----------------------------------------------------------------


# The result of torchsummary.summary()
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [20, 32, 416, 416]             864
#        BatchNorm2d-2         [20, 32, 416, 416]              64
#          LeakyReLU-3         [20, 32, 416, 416]               0
#           CNNBlock-4         [20, 32, 416, 416]               0
#             Conv2d-5         [20, 64, 208, 208]          18,432
#        BatchNorm2d-6         [20, 64, 208, 208]             128
#          LeakyReLU-7         [20, 64, 208, 208]               0
#           CNNBlock-8         [20, 64, 208, 208]               0
#             Conv2d-9         [20, 32, 208, 208]           2,048
#       BatchNorm2d-10         [20, 32, 208, 208]              64
#         LeakyReLU-11         [20, 32, 208, 208]               0
#          CNNBlock-12         [20, 32, 208, 208]               0
#            Conv2d-13         [20, 64, 208, 208]          18,432
#       BatchNorm2d-14         [20, 64, 208, 208]             128
#         LeakyReLU-15         [20, 64, 208, 208]               0
#          CNNBlock-16         [20, 64, 208, 208]               0
#     ResidualBlock-17         [20, 64, 208, 208]               0
#            Conv2d-18        [20, 128, 104, 104]          73,728
#       BatchNorm2d-19        [20, 128, 104, 104]             256
#         LeakyReLU-20        [20, 128, 104, 104]               0
#          CNNBlock-21        [20, 128, 104, 104]               0
#            Conv2d-22         [20, 64, 104, 104]           8,192
#       BatchNorm2d-23         [20, 64, 104, 104]             128
#         LeakyReLU-24         [20, 64, 104, 104]               0
#          CNNBlock-25         [20, 64, 104, 104]               0
#            Conv2d-26        [20, 128, 104, 104]          73,728
#       BatchNorm2d-27        [20, 128, 104, 104]             256
#         LeakyReLU-28        [20, 128, 104, 104]               0
#          CNNBlock-29        [20, 128, 104, 104]               0
#            Conv2d-30         [20, 64, 104, 104]           8,192
#       BatchNorm2d-31         [20, 64, 104, 104]             128
#         LeakyReLU-32         [20, 64, 104, 104]               0
#          CNNBlock-33         [20, 64, 104, 104]               0
#            Conv2d-34        [20, 128, 104, 104]          73,728
#       BatchNorm2d-35        [20, 128, 104, 104]             256
#         LeakyReLU-36        [20, 128, 104, 104]               0
#          CNNBlock-37        [20, 128, 104, 104]               0
#     ResidualBlock-38        [20, 128, 104, 104]               0
#            Conv2d-39          [20, 256, 52, 52]         294,912
#       BatchNorm2d-40          [20, 256, 52, 52]             512
#         LeakyReLU-41          [20, 256, 52, 52]               0
#          CNNBlock-42          [20, 256, 52, 52]               0
#            Conv2d-43          [20, 128, 52, 52]          32,768
#       BatchNorm2d-44          [20, 128, 52, 52]             256
#         LeakyReLU-45          [20, 128, 52, 52]               0
#          CNNBlock-46          [20, 128, 52, 52]               0
#            Conv2d-47          [20, 256, 52, 52]         294,912
#       BatchNorm2d-48          [20, 256, 52, 52]             512
#         LeakyReLU-49          [20, 256, 52, 52]               0
#          CNNBlock-50          [20, 256, 52, 52]               0
#            Conv2d-51          [20, 128, 52, 52]          32,768
#       BatchNorm2d-52          [20, 128, 52, 52]             256
#         LeakyReLU-53          [20, 128, 52, 52]               0
#          CNNBlock-54          [20, 128, 52, 52]               0
#            Conv2d-55          [20, 256, 52, 52]         294,912
#       BatchNorm2d-56          [20, 256, 52, 52]             512
#         LeakyReLU-57          [20, 256, 52, 52]               0
#          CNNBlock-58          [20, 256, 52, 52]               0
#            Conv2d-59          [20, 128, 52, 52]          32,768
#       BatchNorm2d-60          [20, 128, 52, 52]             256
#         LeakyReLU-61          [20, 128, 52, 52]               0
#          CNNBlock-62          [20, 128, 52, 52]               0
#            Conv2d-63          [20, 256, 52, 52]         294,912
#       BatchNorm2d-64          [20, 256, 52, 52]             512
#         LeakyReLU-65          [20, 256, 52, 52]               0
#          CNNBlock-66          [20, 256, 52, 52]               0
#            Conv2d-67          [20, 128, 52, 52]          32,768
#       BatchNorm2d-68          [20, 128, 52, 52]             256
#         LeakyReLU-69          [20, 128, 52, 52]               0
#          CNNBlock-70          [20, 128, 52, 52]               0
#            Conv2d-71          [20, 256, 52, 52]         294,912
#       BatchNorm2d-72          [20, 256, 52, 52]             512
#         LeakyReLU-73          [20, 256, 52, 52]               0
#          CNNBlock-74          [20, 256, 52, 52]               0
#            Conv2d-75          [20, 128, 52, 52]          32,768
#       BatchNorm2d-76          [20, 128, 52, 52]             256
#         LeakyReLU-77          [20, 128, 52, 52]               0
#          CNNBlock-78          [20, 128, 52, 52]               0
#            Conv2d-79          [20, 256, 52, 52]         294,912
#       BatchNorm2d-80          [20, 256, 52, 52]             512
#         LeakyReLU-81          [20, 256, 52, 52]               0
#          CNNBlock-82          [20, 256, 52, 52]               0
#            Conv2d-83          [20, 128, 52, 52]          32,768
#       BatchNorm2d-84          [20, 128, 52, 52]             256
#         LeakyReLU-85          [20, 128, 52, 52]               0
#          CNNBlock-86          [20, 128, 52, 52]               0
#            Conv2d-87          [20, 256, 52, 52]         294,912
#       BatchNorm2d-88          [20, 256, 52, 52]             512
#         LeakyReLU-89          [20, 256, 52, 52]               0
#          CNNBlock-90          [20, 256, 52, 52]               0
#            Conv2d-91          [20, 128, 52, 52]          32,768
#       BatchNorm2d-92          [20, 128, 52, 52]             256
#         LeakyReLU-93          [20, 128, 52, 52]               0
#          CNNBlock-94          [20, 128, 52, 52]               0
#            Conv2d-95          [20, 256, 52, 52]         294,912
#       BatchNorm2d-96          [20, 256, 52, 52]             512
#         LeakyReLU-97          [20, 256, 52, 52]               0
# C:\Users\paulc\.conda\envs\pt3.7\lib\site-packages\torchsummary\torchsummary.py:93: RuntimeWarning: overflow encountered in long_scalars
#   total_output += np.prod(summary[layer]["output_shape"])
#          CNNBlock-98          [20, 256, 52, 52]               0
#            Conv2d-99          [20, 128, 52, 52]          32,768
#      BatchNorm2d-100          [20, 128, 52, 52]             256
#        LeakyReLU-101          [20, 128, 52, 52]               0
#         CNNBlock-102          [20, 128, 52, 52]               0
#           Conv2d-103          [20, 256, 52, 52]         294,912
#      BatchNorm2d-104          [20, 256, 52, 52]             512
#        LeakyReLU-105          [20, 256, 52, 52]               0
#         CNNBlock-106          [20, 256, 52, 52]               0
#    ResidualBlock-107          [20, 256, 52, 52]               0
#           Conv2d-108          [20, 512, 26, 26]       1,179,648
#      BatchNorm2d-109          [20, 512, 26, 26]           1,024
#        LeakyReLU-110          [20, 512, 26, 26]               0
#         CNNBlock-111          [20, 512, 26, 26]               0
#           Conv2d-112          [20, 256, 26, 26]         131,072
#      BatchNorm2d-113          [20, 256, 26, 26]             512
#        LeakyReLU-114          [20, 256, 26, 26]               0
#         CNNBlock-115          [20, 256, 26, 26]               0
#           Conv2d-116          [20, 512, 26, 26]       1,179,648
#      BatchNorm2d-117          [20, 512, 26, 26]           1,024
#        LeakyReLU-118          [20, 512, 26, 26]               0
#         CNNBlock-119          [20, 512, 26, 26]               0
#           Conv2d-120          [20, 256, 26, 26]         131,072
#      BatchNorm2d-121          [20, 256, 26, 26]             512
#        LeakyReLU-122          [20, 256, 26, 26]               0
#         CNNBlock-123          [20, 256, 26, 26]               0
#           Conv2d-124          [20, 512, 26, 26]       1,179,648
#      BatchNorm2d-125          [20, 512, 26, 26]           1,024
#        LeakyReLU-126          [20, 512, 26, 26]               0
#         CNNBlock-127          [20, 512, 26, 26]               0
#           Conv2d-128          [20, 256, 26, 26]         131,072
#      BatchNorm2d-129          [20, 256, 26, 26]             512
#        LeakyReLU-130          [20, 256, 26, 26]               0
#         CNNBlock-131          [20, 256, 26, 26]               0
#           Conv2d-132          [20, 512, 26, 26]       1,179,648
#      BatchNorm2d-133          [20, 512, 26, 26]           1,024
#        LeakyReLU-134          [20, 512, 26, 26]               0
#         CNNBlock-135          [20, 512, 26, 26]               0
#           Conv2d-136          [20, 256, 26, 26]         131,072
#      BatchNorm2d-137          [20, 256, 26, 26]             512
#        LeakyReLU-138          [20, 256, 26, 26]               0
#         CNNBlock-139          [20, 256, 26, 26]               0
#           Conv2d-140          [20, 512, 26, 26]       1,179,648
#      BatchNorm2d-141          [20, 512, 26, 26]           1,024
#        LeakyReLU-142          [20, 512, 26, 26]               0
#         CNNBlock-143          [20, 512, 26, 26]               0
#           Conv2d-144          [20, 256, 26, 26]         131,072
#      BatchNorm2d-145          [20, 256, 26, 26]             512
#        LeakyReLU-146          [20, 256, 26, 26]               0
#         CNNBlock-147          [20, 256, 26, 26]               0
#           Conv2d-148          [20, 512, 26, 26]       1,179,648
#      BatchNorm2d-149          [20, 512, 26, 26]           1,024
#        LeakyReLU-150          [20, 512, 26, 26]               0
#         CNNBlock-151          [20, 512, 26, 26]               0
#           Conv2d-152          [20, 256, 26, 26]         131,072
#      BatchNorm2d-153          [20, 256, 26, 26]             512
#        LeakyReLU-154          [20, 256, 26, 26]               0
#         CNNBlock-155          [20, 256, 26, 26]               0
#           Conv2d-156          [20, 512, 26, 26]       1,179,648
#      BatchNorm2d-157          [20, 512, 26, 26]           1,024
#        LeakyReLU-158          [20, 512, 26, 26]               0
#         CNNBlock-159          [20, 512, 26, 26]               0
#           Conv2d-160          [20, 256, 26, 26]         131,072
#      BatchNorm2d-161          [20, 256, 26, 26]             512
#        LeakyReLU-162          [20, 256, 26, 26]               0
#         CNNBlock-163          [20, 256, 26, 26]               0
#           Conv2d-164          [20, 512, 26, 26]       1,179,648
#      BatchNorm2d-165          [20, 512, 26, 26]           1,024
#        LeakyReLU-166          [20, 512, 26, 26]               0
#         CNNBlock-167          [20, 512, 26, 26]               0
#           Conv2d-168          [20, 256, 26, 26]         131,072
#      BatchNorm2d-169          [20, 256, 26, 26]             512
#        LeakyReLU-170          [20, 256, 26, 26]               0
#         CNNBlock-171          [20, 256, 26, 26]               0
#           Conv2d-172          [20, 512, 26, 26]       1,179,648
#      BatchNorm2d-173          [20, 512, 26, 26]           1,024
#        LeakyReLU-174          [20, 512, 26, 26]               0
#         CNNBlock-175          [20, 512, 26, 26]               0
#    ResidualBlock-176          [20, 512, 26, 26]               0
#           Conv2d-177         [20, 1024, 13, 13]       4,718,592
#      BatchNorm2d-178         [20, 1024, 13, 13]           2,048
#        LeakyReLU-179         [20, 1024, 13, 13]               0
#         CNNBlock-180         [20, 1024, 13, 13]               0
#           Conv2d-181          [20, 512, 13, 13]         524,288
#      BatchNorm2d-182          [20, 512, 13, 13]           1,024
#        LeakyReLU-183          [20, 512, 13, 13]               0
#         CNNBlock-184          [20, 512, 13, 13]               0
#           Conv2d-185         [20, 1024, 13, 13]       4,718,592
#      BatchNorm2d-186         [20, 1024, 13, 13]           2,048
#        LeakyReLU-187         [20, 1024, 13, 13]               0
#         CNNBlock-188         [20, 1024, 13, 13]               0
#           Conv2d-189          [20, 512, 13, 13]         524,288
#      BatchNorm2d-190          [20, 512, 13, 13]           1,024
#        LeakyReLU-191          [20, 512, 13, 13]               0
#         CNNBlock-192          [20, 512, 13, 13]               0
#           Conv2d-193         [20, 1024, 13, 13]       4,718,592
#      BatchNorm2d-194         [20, 1024, 13, 13]           2,048
#        LeakyReLU-195         [20, 1024, 13, 13]               0
#         CNNBlock-196         [20, 1024, 13, 13]               0
#           Conv2d-197          [20, 512, 13, 13]         524,288
#      BatchNorm2d-198          [20, 512, 13, 13]           1,024
#        LeakyReLU-199          [20, 512, 13, 13]               0
#         CNNBlock-200          [20, 512, 13, 13]               0
#           Conv2d-201         [20, 1024, 13, 13]       4,718,592
#      BatchNorm2d-202         [20, 1024, 13, 13]           2,048
#        LeakyReLU-203         [20, 1024, 13, 13]               0
#         CNNBlock-204         [20, 1024, 13, 13]               0
#           Conv2d-205          [20, 512, 13, 13]         524,288
#      BatchNorm2d-206          [20, 512, 13, 13]           1,024
#        LeakyReLU-207          [20, 512, 13, 13]               0
#         CNNBlock-208          [20, 512, 13, 13]               0
#           Conv2d-209         [20, 1024, 13, 13]       4,718,592
#      BatchNorm2d-210         [20, 1024, 13, 13]           2,048
#        LeakyReLU-211         [20, 1024, 13, 13]               0
#         CNNBlock-212         [20, 1024, 13, 13]               0
#    ResidualBlock-213         [20, 1024, 13, 13]               0
#           Conv2d-214          [20, 512, 13, 13]         524,288
#      BatchNorm2d-215          [20, 512, 13, 13]           1,024
#        LeakyReLU-216          [20, 512, 13, 13]               0
#         CNNBlock-217          [20, 512, 13, 13]               0
#           Conv2d-218         [20, 1024, 13, 13]       4,718,592
#      BatchNorm2d-219         [20, 1024, 13, 13]           2,048
#        LeakyReLU-220         [20, 1024, 13, 13]               0
#         CNNBlock-221         [20, 1024, 13, 13]               0
#           Conv2d-222          [20, 512, 13, 13]         524,288
#      BatchNorm2d-223          [20, 512, 13, 13]           1,024
#        LeakyReLU-224          [20, 512, 13, 13]               0
#         CNNBlock-225          [20, 512, 13, 13]               0
#           Conv2d-226         [20, 1024, 13, 13]       4,718,592
#      BatchNorm2d-227         [20, 1024, 13, 13]           2,048
#        LeakyReLU-228         [20, 1024, 13, 13]               0
#         CNNBlock-229         [20, 1024, 13, 13]               0
#    ResidualBlock-230         [20, 1024, 13, 13]               0
#           Conv2d-231          [20, 512, 13, 13]         524,288
#      BatchNorm2d-232          [20, 512, 13, 13]           1,024
#        LeakyReLU-233          [20, 512, 13, 13]               0
#         CNNBlock-234          [20, 512, 13, 13]               0
#           Conv2d-235         [20, 1024, 13, 13]       4,718,592
#      BatchNorm2d-236         [20, 1024, 13, 13]           2,048
#        LeakyReLU-237         [20, 1024, 13, 13]               0
#         CNNBlock-238         [20, 1024, 13, 13]               0
#           Conv2d-239           [20, 24, 13, 13]          24,600
#         CNNBlock-240           [20, 24, 13, 13]               0
#  ScalePrediction-241         [20, 3, 13, 13, 8]               0
#           Conv2d-242          [20, 256, 13, 13]         131,072
#      BatchNorm2d-243          [20, 256, 13, 13]             512
#        LeakyReLU-244          [20, 256, 13, 13]               0
#         CNNBlock-245          [20, 256, 13, 13]               0
#         Upsample-246          [20, 256, 26, 26]               0
#           Conv2d-247          [20, 256, 26, 26]         196,608
#      BatchNorm2d-248          [20, 256, 26, 26]             512
#        LeakyReLU-249          [20, 256, 26, 26]               0
#         CNNBlock-250          [20, 256, 26, 26]               0
#           Conv2d-251          [20, 512, 26, 26]       1,179,648
#      BatchNorm2d-252          [20, 512, 26, 26]           1,024
#        LeakyReLU-253          [20, 512, 26, 26]               0
#         CNNBlock-254          [20, 512, 26, 26]               0
#           Conv2d-255          [20, 256, 26, 26]         131,072
#      BatchNorm2d-256          [20, 256, 26, 26]             512
#        LeakyReLU-257          [20, 256, 26, 26]               0
#         CNNBlock-258          [20, 256, 26, 26]               0
#           Conv2d-259          [20, 512, 26, 26]       1,179,648
#      BatchNorm2d-260          [20, 512, 26, 26]           1,024
#        LeakyReLU-261          [20, 512, 26, 26]               0
#         CNNBlock-262          [20, 512, 26, 26]               0
#    ResidualBlock-263          [20, 512, 26, 26]               0
#           Conv2d-264          [20, 256, 26, 26]         131,072
#      BatchNorm2d-265          [20, 256, 26, 26]             512
#        LeakyReLU-266          [20, 256, 26, 26]               0
#         CNNBlock-267          [20, 256, 26, 26]               0
#           Conv2d-268          [20, 512, 26, 26]       1,179,648
#      BatchNorm2d-269          [20, 512, 26, 26]           1,024
#        LeakyReLU-270          [20, 512, 26, 26]               0
#         CNNBlock-271          [20, 512, 26, 26]               0
#           Conv2d-272           [20, 24, 26, 26]          12,312
#         CNNBlock-273           [20, 24, 26, 26]               0
#  ScalePrediction-274         [20, 3, 26, 26, 8]               0
#           Conv2d-275          [20, 128, 26, 26]          32,768
#      BatchNorm2d-276          [20, 128, 26, 26]             256
#        LeakyReLU-277          [20, 128, 26, 26]               0
#         CNNBlock-278          [20, 128, 26, 26]               0
#         Upsample-279          [20, 128, 52, 52]               0
#           Conv2d-280          [20, 128, 52, 52]          49,152
#      BatchNorm2d-281          [20, 128, 52, 52]             256
#        LeakyReLU-282          [20, 128, 52, 52]               0
#         CNNBlock-283          [20, 128, 52, 52]               0
#           Conv2d-284          [20, 256, 52, 52]         294,912
#      BatchNorm2d-285          [20, 256, 52, 52]             512
#        LeakyReLU-286          [20, 256, 52, 52]               0
#         CNNBlock-287          [20, 256, 52, 52]               0
#           Conv2d-288          [20, 128, 52, 52]          32,768
#      BatchNorm2d-289          [20, 128, 52, 52]             256
#        LeakyReLU-290          [20, 128, 52, 52]               0
#         CNNBlock-291          [20, 128, 52, 52]               0
#           Conv2d-292          [20, 256, 52, 52]         294,912
#      BatchNorm2d-293          [20, 256, 52, 52]             512
#        LeakyReLU-294          [20, 256, 52, 52]               0
#         CNNBlock-295          [20, 256, 52, 52]               0
#    ResidualBlock-296          [20, 256, 52, 52]               0
#           Conv2d-297          [20, 128, 52, 52]          32,768
#      BatchNorm2d-298          [20, 128, 52, 52]             256
#        LeakyReLU-299          [20, 128, 52, 52]               0
#         CNNBlock-300          [20, 128, 52, 52]               0
#           Conv2d-301          [20, 256, 52, 52]         294,912
#      BatchNorm2d-302          [20, 256, 52, 52]             512
#        LeakyReLU-303          [20, 256, 52, 52]               0
#         CNNBlock-304          [20, 256, 52, 52]               0
#           Conv2d-305           [20, 24, 52, 52]           6,168
#         CNNBlock-306           [20, 24, 52, 52]               0
#  ScalePrediction-307         [20, 3, 52, 52, 8]               0
# ================================================================
# Total params: 61,534,504
# Trainable params: 61,534,504
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 39.61
# Forward/backward pass size (MB): 8276.82
# Params size (MB): 234.74
# Estimated Total Size (MB): 8551.17
# ----------------------------------------------------------------



# The result of directly print(model)
# YOLOv3(
#   (layers): ModuleList(
#     (0): CNNBlock(
#       (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (1): CNNBlock(
#       (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (2): ResidualBlock(
#       (layers): ModuleList(
#         (0): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#       )
#     )
#     (3): CNNBlock(
#       (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (4): ResidualBlock(
#       (layers): ModuleList(
#         (0): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (1): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#       )
#     )
#     (5): CNNBlock(
#       (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (6): ResidualBlock(
#       (layers): ModuleList(
#         (0): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (1): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (2): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (3): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (4): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (5): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (6): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (7): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#       )
#     )
#     (7): CNNBlock(
#       (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (8): ResidualBlock(
#       (layers): ModuleList(
#         (0): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (1): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (2): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (3): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (4): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (5): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (6): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (7): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#       )
#     )
#     (9): CNNBlock(
#       (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (10): ResidualBlock(
#       (layers): ModuleList(
#         (0): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (1): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (2): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#         (3): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#       )
#     )
#     (11): CNNBlock(
#       (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (12): CNNBlock(
#       (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (13): ResidualBlock(
#       (layers): ModuleList(
#         (0): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#       )
#     )
#     (14): CNNBlock(
#       (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (15): ScalePrediction(
#       (pred): Sequential(
#         (0): CNNBlock(
#           (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (leaky): LeakyReLU(negative_slope=0.1)
#         )
#         (1): CNNBlock(
#           (conv): Conv2d(1024, 24, kernel_size=(1, 1), stride=(1, 1))
#           (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (leaky): LeakyReLU(negative_slope=0.1)
#         )
#       )
#     )
#     (16): CNNBlock(
#       (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (17): Upsample(scale_factor=2.0, mode=nearest)
#     (18): CNNBlock(
#       (conv): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (19): CNNBlock(
#       (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (20): ResidualBlock(
#       (layers): ModuleList(
#         (0): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#       )
#     )
#     (21): CNNBlock(
#       (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (22): ScalePrediction(
#       (pred): Sequential(
#         (0): CNNBlock(
#           (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (leaky): LeakyReLU(negative_slope=0.1)
#         )
#         (1): CNNBlock(
#           (conv): Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1))
#           (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (leaky): LeakyReLU(negative_slope=0.1)
#         )
#       )
#     )
#     (23): CNNBlock(
#       (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (24): Upsample(scale_factor=2.0, mode=nearest)
#     (25): CNNBlock(
#       (conv): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (26): CNNBlock(
#       (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (27): ResidualBlock(
#       (layers): ModuleList(
#         (0): Sequential(
#           (0): CNNBlock(
#             (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#           (1): CNNBlock(
#             (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (leaky): LeakyReLU(negative_slope=0.1)
#           )
#         )
#       )
#     )
#     (28): CNNBlock(
#       (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (leaky): LeakyReLU(negative_slope=0.1)
#     )
#     (29): ScalePrediction(
#       (pred): Sequential(
#         (0): CNNBlock(
#           (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#           (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (leaky): LeakyReLU(negative_slope=0.1)
#         )
#         (1): CNNBlock(
#           (conv): Conv2d(256, 24, kernel_size=(1, 1), stride=(1, 1))
#           (bn): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (leaky): LeakyReLU(negative_slope=0.1)
#         )
#       )
#     )
#   )
# )


