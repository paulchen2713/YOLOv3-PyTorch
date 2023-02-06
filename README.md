# YOLOv3-PyTorch
- The original code was copied from [YOLOv3-PyTorch](https://github.com/SannaPersson/YOLOv3-PyTorch)
- 2023.02.06
  - On lab PC, create a new env ```pt3.7``` through command ```conda create --name pt3.7 python=3.7```
    - To use the env ```conda activate pt3.7```
    - To leave the env ```conda deactivate```
    - Actual env and pkgs locates at ```C:\Users\Paul\.conda\envs\pt3.7```, don't know why it is not been stored in ```D Drive```
  - Upgrade default conda env ```base``` through command ```conda update -n base -c defaults conda```
    <img src='https://i.imgur.com/6JozbHB.png' width=90% height=90%>
    - It has to be done under ```(base) C:\Windows\system32>```
    <img src='https://i.imgur.com/cYJM1XN.png' width=80% height=80%>
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
    - Actual dependency, new requirements is:
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
- 2023.02.05
  - first download the YOLOv3 weights from https://pjreddie.com/media/files/yolov3.weights as ```yolov3.weights``` and put it at the same directory
  - then run ```model_with_weights2.py```, it will save the weights to PyTorch format. we name the output weights as ```checkpoint-2023-02-05.pth.tar``` also in the same directory
  - inside the directory
    ![image](https://user-images.githubusercontent.com/95068443/216808211-7a95bcdf-4444-4116-965b-6462cb20646a.png)
  - I override most of the files with my previous ones, except for ```model_with_weights2.py```

  
