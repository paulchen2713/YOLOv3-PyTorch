# YOLOv3-PyTorch
## Reference
- The original code was copied from [YOLOv3-PyTorch](https://github.com/SannaPersson/YOLOv3-PyTorch)

## Notes
- 2023.02.21
  - modified from YOLO-CFAR
    ```clike
    (pt3.8) D:\Datasets\YOLOv3-PyTorch\YOLOv3-debug1>D:/ProgramData/Anaconda3/envs/pt3.8/python.exe d:/Datasets/YOLOv3-PyTorch/YOLOv3-debug1/train.py
      0%|                                                                                                                                            | 0/375 [00:03<?, ?it/s]
    Traceback (most recent call last):
      File "d:/Datasets/YOLOv3-PyTorch/YOLOv3-debug1/train.py", line 166, in <module>
        main()
      File "d:/Datasets/YOLOv3-PyTorch/YOLOv3-debug1/train.py", line 107, in main    
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)    
      File "d:/Datasets/YOLOv3-PyTorch/YOLOv3-debug1/train.py", line 57, in train_fn
        out = model(x)
      File "D:\ProgramData\Anaconda3\envs\pt3.8\lib\site-packages\torch\nn\modules\module.py", line 1194, in _call_impl
        return forward_call(*input, **kwargs)
      File "d:\Datasets\YOLOv3-PyTorch\YOLOv3-debug1\model.py", line 191, in forward
        x = layer(x) #
      File "D:\ProgramData\Anaconda3\envs\pt3.8\lib\site-packages\torch\nn\modules\module.py", line 1194, in _call_impl
        return forward_call(*input, **kwargs)
      File "d:\Datasets\YOLOv3-PyTorch\YOLOv3-debug1\model.py", line 110, in forward
        return self.leaky(self.bn(self.conv(x))) # bn_act()
      File "D:\ProgramData\Anaconda3\envs\pt3.8\lib\site-packages\torch\nn\modules\module.py", line 1194, in _call_impl
        return forward_call(*input, **kwargs)
      File "D:\ProgramData\Anaconda3\envs\pt3.8\lib\site-packages\torch\nn\modules\conv.py", line 463, in forward
        return self._conv_forward(input, self.weight, self.bias)
      File "D:\ProgramData\Anaconda3\envs\pt3.8\lib\site-packages\torch\nn\modules\conv.py", line 459, in _conv_forward
        return F.conv2d(input, weight, bias, self.stride,
    RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[16, 256, 64, 3] to have 3 channels, but got 256 channels instead
    ```
  - modified from YOLO-Pascal_VOC
    ```clike
    (pt3.8) D:\Datasets\YOLOv3-PyTorch\YOLOv3-debug2>D:/ProgramData/Anaconda3/envs/pt3.8/python.exe d:/Datasets/YOLOv3-PyTorch/YOLOv3-debug2/train.py
      0%|                                                                                                                           | 0/5999 [00:00<?, ?it/s]
    x:  torch.Size([1, 3, 256, 64])
    y0: torch.Size([1, 3, 2, 2, 6])
    y1: torch.Size([1, 3, 2, 2, 6])
    y2: torch.Size([1, 3, 2, 2, 6])
      0%|                                                                                                                           | 0/5999 [00:04<?, ?it/s]
    Traceback (most recent call last):
      File "d:/Datasets/YOLOv3-PyTorch/YOLOv3-debug2/train.py", line 144, in <module>
        main()
      File "d:/Datasets/YOLOv3-PyTorch/YOLOv3-debug2/train.py", line 91, in main
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
      File "d:/Datasets/YOLOv3-PyTorch/YOLOv3-debug2/train.py", line 47, in train_fn
        loss_fn(out[0], y0, scaled_anchors[0])
      File "D:\ProgramData\Anaconda3\envs\pt3.8\lib\site-packages\torch\nn\modules\module.py", line 1194, in _call_impl
        return forward_call(*input, **kwargs)
      File "d:\Datasets\YOLOv3-PyTorch\YOLOv3-debug2\loss.py", line 83, in forward
        no_object_loss = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),)
    IndexError: The shape of the mask [1, 3, 2, 2] at index 2 does not match the shape of the indexed tensor [1, 3, 8, 2, 1] at index 2
    ```
- 2023.02.20
  - We now have the model trained on ```Pascal_VOC``` dataset with the following result
    ![](https://i.imgur.com/mZN3b25.png)
  - The model was evaluated with confidence ```0.6``` and IOU threshold ```0.5``` using NMS
    |          Model          |     mAP_50    |
    | ----------------------- | ------------- |
    | ```YOLOv3-Pascal_VOC``` | ```75.7776``` |
    - The overlapped area means <img src = 'https://i.imgur.com/SHNltVr.png' height=10% width=10% >
    - IoU threshold value to the actual overlapped area
    <img src = 'https://i.imgur.com/quULxhX.png' height=70% width=70% >
- 2023.02.18
  - The virtual envs are summarized below:
  - My PC ```(Intel i7-8700 + Nvidia Geforce RTX 2060)```: 
    - env ```pt3.7``` with CUDA 
        ```python
        python==3.7.13
        numpy==1.19.2
        pytorch==1.7.1
        torchaudio==0.7.2
        torchvision==0.8.2
        pandas==1.2.1
        pillow==8.1.0 
        tqdm==4.56.0
        matplotlib==3.3.4
        albumentations==0.5.2
        ```
  - Lab PC ```(Intel i7-12700 + Nvidia Geforce RTX 3060 Ti)```: 
    - env ```pt3.7``` without CUDA
        ```python
        python==3.7.13
        numpy==1.21.6
        torch==1.13.1
        torchvision==0.14.1
        pandas==1.3.5
        pillow==9.4.0
        tqdm==4.64.1
        matplotlib==3.5.3
        albumentations==1.3.0
        ```
    - env ```pt3.8``` with CUDA
        ```python
        python==3.8.16
        numpy==1.23.5
        pytorch==1.13.1
        pytorch-cuda==11.7
        torchaudio==0.13.1             
        torchvision==0.14.1
        pandas==1.5.2
        pillow==9.3.0
        tqdm==4.64.1
        matplotlib==3.6.2
        albumentations==1.3.0
        ```
  - An annoying bug in ```dataset.py``` due to pytorch version
    - The code segment that contains potential bug (on line ```149``` and ```155```)
    ![](https://i.imgur.com/w5hUN05.png)
    ![](https://i.imgur.com/R7TKmAo.png)
    - ```scale_idx = anchor_idx // self.num_anchors_per_scale``` works fine on my PC, but on lab PC will get the following warning, so I naturally followed the suggestions and changed the syntax to ([```torch.div()```](https://pytorch.org/docs/stable/generated/torch.div.html))
        ```clike!
        UserWarning: __floordiv__ is deprecated, and its behavior will change in a future version of pytorch.
        ```
    - After following the suggestion and chage  the deprecated usage ```//``` we have: ```scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='floor')```. This piece of code works fine on lab PC, under both env ```pt3.7``` and ```pt3.8```, but failed on my PC.
    - The error only occur on my PC, under env ```pt3.7```, but this env is the initial and stable one.
        ```clike
        Original Traceback (most recent call last):
          File "C:\Users\paulc\.conda\envs\pt3.7\lib\site-packages\torch\utils\data\_utils\worker.py", line 198, in _worker_loop
            data = fetcher.fetch(index)
          File "C:\Users\paulc\.conda\envs\pt3.7\lib\site-packages\torch\utils\data\_utils\fetch.py", line 44, in fetch
            data = [self.dataset[idx] for idx in possibly_batched_index]
          File "C:\Users\paulc\.conda\envs\pt3.7\lib\site-packages\torch\utils\data\_utils\fetch.py", line 44, in <listcomp>
            data = [self.dataset[idx] for idx in possibly_batched_index]
          File "d:\Datasets\YOLOv3-PyTorch\dataset.py", line 153, in __getitem__
            scale_idx = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='floor')
        TypeError: div() got an unexpected keyword argument 'rounding_mode'
        ```
- 2023.02.10
  - Trying newer stable PyTorch and CUDA version for the project
  - Python 3.8 + CUDA 11.7 
    - ```conda create --name pt3.8 python=3.8```
    - ```conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia``` ([Install PyTorch](https://pytorch.org/))
  - **Interesting to know!** 
    - 如果透過系統管理員開啟 ```Anaconda Prompt``` 並安裝的環境會存在 D 槽的 ```D:/ProgramData/Anaconda3/envs/```
    ![](https://i.imgur.com/G979f4e.png)
    - 反之，直接開 ```Anaconda Prompt``` 安裝的環境會存在 C 槽的 ```C:/Users/Paul/.conda/envs/```
    - 以後記得都用系統管理員執行!
  - The new dependency is:
    ```python
    numpy==1.23.5
    matplotlib==3.6.2
    pytorch==1.13.1
    pytorch-cuda==11.7
    torchaudio==0.13.1             
    torchvision==0.14.1
    tqdm==4.64.1
    albumentations==1.3.0
    pandas==1.5.2
    pillow==9.3.0
    ```
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
  - Find a newer stable PyTorch and CUDA version for the project
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

  
