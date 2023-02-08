# YOLOv3-PyTorch
## reference
- The original code was copied from [YOLOv3-PyTorch](https://github.com/SannaPersson/YOLOv3-PyTorch)


## notes
- 2023.02.08
  - The ```YOLOv3``` model is trainable with ```Pascal_VOC``` dataset
    - But it's bind with ```Albumentations``` / data augmentations, which means we need to decoupling it
  - To our knowledge, we know that **pre-training is good for our task**, least that's what the paper says, so I was trying to solve this issue
    - C. Decourt, R. VanRullen, D. Salle and T. Oberlin, "[DAROD: A Deep Automotive Radar Object Detector on Range-Doppler maps](https://ieeexplore.ieee.org/abstract/document/9827281)," *2022 IEEE Intelligent Vehicles Symposium (IV)*, Aachen, Germany, 2022, pp. 112-118.
  - Originally, I want to convert the pre-trained weights from darknet_format to pytorch_format, **it does not work**
  - Add two additional functions ```load_CNN_weights()``` and ```load_darknet_weights()``` in ```model.py``` to read the darknet weights
    - fun fact, there are in total ```62001757``` parameters of YOLOv3
- At least, in the future, we can separate our training process if needed
  - we can "save checkpoint" for every epoch or every 10, 20 epochs
  - **but the correctness of doing so is unsure**, what I mean unsure is that say we already train for 100 epochs and achieve centain level of preformance, but if we stop and continue the training for another 100 epochs, the performance may drop
  - **remember** to test it with ```seed_everything()``` and make sure it works
- Need to find a newer dependency
  - **Currently run without CUDA** support since there will be PyTorch 2.0 updates soon
  - [Deprecation of CUDA 11.6 and Python 3.7 Support](https://pytorch.org/blog/deprecation-cuda-python-support/?utm_content=236805635&utm_medium=social&utm_source=linkedin&hss_channel=lcp-78618366)
    - Please note that as of **Feb 1**, **CUDA 11.6 and Python 3.7 are no longer included** in the Stable CUDA
  - There is a new paper that says their model can learn spatial and temporal relationships between frames by leveraging the characteristics of the FMCW radar signal
    - Decourt, Colin, et al. "[A recurrent CNN for online object detection on raw radar frames](https://arxiv.org/abs/2212.11172)." *arXiv preprint arXiv:2212.11172* (2022).
  - The comparison between different generations showed that, though newer versions of the model may be more complex, they are not necessarily bigger
    - YOLOv3 ```222``` layers, ```62001757``` parameters
    - YOLOv4 ```488``` layers, ```64363101``` parameters
      - YOLOv4-CSP ```516``` layers, ```52921437``` parameters
    - YOLOv7 ```314``` layers, ```36907898``` parameters
- Future works
  - Make sure we can properly run ```train.py``` with radar dataset
  - Find a proper way to measure the "communication overhead"
  - Test the functionality of  ```seed_everything()```, check if it works like the way we think
- 2023.02.07
  - The code ```detect.py``` and ```model_with_weights2.py``` works fine, but the result may not be the way as we expected
  - Need to figure out the usability of the converted weights, since there is a huge difference between random weights and the converted weights, maybe it's not complete garbage
- 2023.02.06
  - On lab PC, create a new env ```pt3.7``` through command ```conda create --name pt3.7 python=3.7```
    - to use the env ```conda activate pt3.7```
    - to leave the env ```conda deactivate```
    - actual env and pkgs locates at ```C:\Users\Paul\.conda\envs\pt3.7```, don't know why it is not been stored in ```D Drive```
  - Upgrade default conda env ```base``` through command ```conda update -n base -c defaults conda```
    - It has to be done under ```(base) C:\Windows\system32>```
  - Install all the packages through ```pip install -r requirements.txt```
    - content in the requirements file
        ```python
        numpy>=1.19.2
        matplotlib>=3.3.4
        torch>=1.7.1
        tqdm>=4.56.0
        torchvision>=0.8.2
        albumentations>=0.5.2
        pandas>=1.2.1
        Pillow>=8.1.0
        ```
    - cmd output stored as ```D:/Datasets/YOLOv3-PyTorch/logs/installation_logs_0206.txt```
    - actual dependency, the new requirement is:
        ```python
        numpy==1.21.6
        matplotlib==3.5.3
        torch==1.13.1
        tqdm==4.64.1
        torchvision==0.14.1
        albumentations==1.3.0
        pandas==1.3.5
        Pillow==9.4.0
        ```
  - Currently run without CUDA support since there will be PyTorch 2.0 updates soon
    - [Deprecation of CUDA 11.6 and Python 3.7 Support](https://pytorch.org/blog/deprecation-cuda-python-support/?utm_content=236805635&utm_medium=social&utm_source=linkedin&hss_channel=lcp-78618366)
    - **Please note that as of Feb 1, CUDA 11.6 and Python 3.7 are no longer included**
  - Run ```model_with_weights2.py``` again on lab PC to generate the weights in PyTorch format
    - we name the output weights as ```checkpoint-2023-02-06.pth.tar``` also stored in the same directory
    ![](https://i.imgur.com/WAncq96.png)
  - Wanted to test the training ability using ```PASCAL_VOC``` dataset
    - download the preprocessed ```PASCAL_VOC``` dataset from [kaggle](https://www.kaggle.com/datasets/aladdinpersson/pascal-voc-dataset-used-in-yolov3-video)
    - download the preprocessed ```MS-COCO``` dataset from [kaggle](https://www.kaggle.com/datasets/79abcc2659dc745fddfba1864438afb2fac3fabaa5f37daa8a51e36466db101e)
  - But first, we have to test the converted weights to check if they actually work
    - to do so, maybe we could write a program ```detect.py``` and test the weights with some inference samples
    - if it can predict perfectly, then we may assume it is converted correctly
    - Okay, it does not work..., the inference outputs are bunch of random tags
- 2023.02.05
  - first download the YOLOv3 weights from https://pjreddie.com/media/files/yolov3.weights as ```yolov3.weights``` and put it at the same directory
  - then run ```model_with_weights2.py```, it will save the weights to PyTorch format. we name the output weights as ```checkpoint-2023-02-05.pth.tar``` also in the same directory
  - inside the directory
    ![image](https://user-images.githubusercontent.com/95068443/216808211-7a95bcdf-4444-4116-965b-6462cb20646a.png)
  - I override most of the files with my previous ones, except for ```model_with_weights2.py```

  
