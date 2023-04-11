# YOLOv3-PyTorch

## Notes
- 2023.04.10
  - Need to recompute / regenerate anchors for YOLO [Training YOLO? Select Anchor Boxes Like This](https://towardsdatascience.com/training-yolo-select-anchor-boxes-like-this-3226cb8d7f0b)
  - for YOLOv2 ```AlexeyAB/darknet/scripts/``` [```gen_anchors.py```](https://github.com/AlexeyAB/darknet/blob/master/scripts/gen_anchors.py)
    - The anchor boxes were calculated with a k-means clustering algorithm only
    - With ```1 - IoU``` as a distance metric
    - Doing k-means clustering only is a good approach already
  - for YOLOv5 / YOLOv7 ```ultralytics/yolov5/utils/``` [```autoanchor.py```](https://github.com/ultralytics/yolov5/blob/master/utils/autoanchor.py)
  - ultralytics YOLOv5 Docs [Train Custom Data](https://docs.ultralytics.com/yolov5/train_custom_data/)
- Auto-anchor algorithm
  - ```Step 0.``` K-means (with simple Euclidean distance) is used to get the initial guess for anchor boxes
    - We also can do it with ```1 - IoU``` as a distance metric
  - ```Step 1.``` Get bounding box sizes from the train data
  - ```Step 2.``` Choose a metric to define anchor fitness
    - Ideally, the metric should be connected to the loss function
  - ```Step 3.``` Do clustering to get an initial guess for anchors
  - ```Step 4.``` Evolve anchors to improve anchor fitness
- Things I'm Googling but haven't finished reading
  - Faster RCNN with PyTorch
    - PyTorch Docs [TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
    - PyTorch Docs [MODELS AND PRE-TRAINED WEIGHTS](https://pytorch.org/vision/main/models.html)
    - PyTorch Source Code [```fasterrcnn_resnet50_fpn()```](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html#fasterrcnn_resnet50_fpn)
    - 知呼 FasterRCNN 解析 [pytorch官方FasterRCNN代碼](https://zhuanlan.zhihu.com/p/145842317)
  - Faster RCNN reproduction
    - Kaggle object detection [Aquarium Dataset](https://www.kaggle.com/datasets/sharansmenon/aquarium-dataset)
    - Kaggle Pytorch Starter -  [FasterRCNN Train](https://www.kaggle.com/code/pestipeti/pytorch-starter-fasterrcnn-train/notebook) 
    - github search for [faster-r-cnn](https://github.com/search?q=faster-r-cnn&type=repositories&p=5)
  - Kmeans implementation
    - scikit-learn [Clustering with kmeans](https://scikit-learn.org/stable/modules/clustering.html#k-means)
    - scikit-learn [Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)
    - scikit-learn [```sklearn.cluster.KMeans()```](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
    - Tech-with-Tim [Implementing K Means Clustering](https://www.techwithtim.net/tutorials/machine-learning-python/k-means-2/) 
    ![](https://i.imgur.com/bgqHKHr.png)
    - Sentdex [K-Means from Scratch in Python](https://pythonprogramming.net/k-means-from-scratch-machine-learning-tutorial/)
    ![](https://i.imgur.com/FVv5sMX.png)
- 2023.04.09
  - 過去 10 天確診啥也沒做
- 2023.03.28
  - I tried to train the model until a point where we're satisfied with its performance, then we can do the edge computing modifications on it
  - Quick recap:
    - The [DAROD paper](https://ieeexplore.ieee.org/document/9827281) propose a light architecture for the ```Faster R-CNN``` object detector on this particular task
    - They can reach respectively an ```mAP@0.5``` and ```mAP@0.3``` of ```55.83``` and ```70.68```
    - So our goal is to at least get a better mAP then they did
  - The current ```mAP@50``` (for every ```100``` epochs) and ```mean loss``` (for every epoch), for a total of ```300``` epochs of training:
    ![](https://i.imgur.com/4xcdypl.png)
    ```
    max training loss (on average): 20.516442289352415
    min training loss (on average): 1.0732185713450113
    ```
  - To further analyze where the problems are, I first extracted some of the data that I think might be helpful 
  - The file tree structure:
    ```
    D:/Datasets/RADA/RD_JPG/training_logs>tree
    D:.
    ├─mAP
    ├─test
    │  ├─class_accuracy
    │  ├─no_object_accuracy
    │  └─object_accuracy
    └─train
        ├─class_accuracy
        ├─losses
        ├─mean_loss
        ├─no_object_accuracy
        └─object_accuracy
    ```
  - Some other results
    - train-class-accuracy vs. test-class-accuracy
        ![](https://i.imgur.com/8I03exy.png)
    - train-no-object-accuracy vs. test-no-object-accuracy
        ![](https://i.imgur.com/dXqZ2Ft.png)
    - train-object-accuracy vs. test-object-accuracy
        ![](https://i.imgur.com/HADZnUl.png)
        ```
        min training accuracy: 2.3661680221557617
        max training accuracy: 94.16690826416016

        min testing accuracy: 46.69877624511719
        max testing accuracy: 72.34597778320312
        ```
  - The layers of the model
    ```python
    layer 0:  torch.Size([20, 32, 416, 416])
    layer 1:  torch.Size([20, 64, 208, 208])
    layer 2:  torch.Size([20, 64, 208, 208])
    layer 3:  torch.Size([20, 128, 104, 104])
    layer 4:  torch.Size([20, 128, 104, 104])
    layer 5:  torch.Size([20, 256, 52, 52])
    layer 6:  torch.Size([20, 256, 52, 52])
    layer 7:  torch.Size([20, 512, 26, 26])
    layer 8:  torch.Size([20, 512, 26, 26])
    layer 9:  torch.Size([20, 1024, 13, 13])
    layer 10:  torch.Size([20, 1024, 13, 13])
    layer 11:  torch.Size([20, 512, 13, 13])
    layer 12:  torch.Size([20, 1024, 13, 13])
    layer 13:  torch.Size([20, 1024, 13, 13])
    layer 14:  torch.Size([20, 512, 13, 13])
    layer 16:  torch.Size([20, 256, 13, 13])
    layer 17:  torch.Size([20, 256, 26, 26])
    layer 18:  torch.Size([20, 256, 26, 26])
    layer 19:  torch.Size([20, 512, 26, 26])
    layer 20:  torch.Size([20, 512, 26, 26])
    layer 21:  torch.Size([20, 256, 26, 26])
    layer 23:  torch.Size([20, 128, 26, 26])
    layer 24:  torch.Size([20, 128, 52, 52])
    layer 25:  torch.Size([20, 128, 52, 52])
    layer 26:  torch.Size([20, 256, 52, 52])
    layer 27:  torch.Size([20, 256, 52, 52])
    layer 28:  torch.Size([20, 128, 52, 52])
    ```
    ```python
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
    ]
    ```
- 2023.03.19
  - The actual size of each input image is: 
    - ```875-by-1489``` or ```310-by-1240```
    ![](https://i.imgur.com/Cjg1AiQ.png)
  - The resizing results are completely different. We could even conclude that they are wrong (and I don't know why), since we might not need to resize images anymore. Currently, I am just ignoring this issue
  ![](https://i.imgur.com/QoYz9TP.jpg)
  ![](https://i.imgur.com/wexdIfa.png)
  - Some samples of person, cyclist and car:
    ![](https://i.imgur.com/2axeJNC.jpg)
    ![](https://i.imgur.com/w3ivjzM.jpg)
    ![](https://i.imgur.com/bbIMrqB.jpg)
  - I first tried to run ```train.py``` for ```100``` epochs with the following config settings:
    ![](https://i.imgur.com/M2u8BtK.png)
  - The resulted ```mAP``` is ```0.182485```
    ![](https://i.imgur.com/eGgLni0.png)
    ![](https://i.imgur.com/faPMJob.png)
    - The code for extracting the data from the log files ```read_logs.py```
      ![](https://i.imgur.com/yzPEsUE.png)
      ![](https://i.imgur.com/CEeq9LO.png)
      ![](https://i.imgur.com/EGTWlkP.png)
      ![](https://i.imgur.com/XIehPQV.png)
- 2023.03.16
  - It's finally trainable now
    ![](https://i.imgur.com/Q8anAyR.png)
  - The major mistakes that I made were: Misinterpreting the labels, but actually translating them correctly.
    - In short, simply switching the ```x``` and ```y``` coordinates will solve our problems
    - This makes me wonder, How did I get it right when replicating ```YOLO-CFAR``` before?
    - Since the shape of the feature map is printed as ```torch.Size([256, 64, 3])```, it shows the same coordinate system as the ```RD map``` where the origin ```(0, 0)``` is located at the top left corner
    - But it turns out that's not the case. The model still recognizes the bottom left corner as the origin, which is the same as we usually do.
  - The correct way to translate the labels
    ![](https://i.imgur.com/DoAn99t.png)
    ![](https://i.imgur.com/skwA1D3.png)
- 2023.03.15
  - Still not actually trainable
    ```clike!
    ValueError: Expected x_min for bbox (-0.103515625, 0.306640625, 0.224609375, 0.365234375, 2.0) to be in the range [0.0, 1.0], got -0.103515625.
    ```
    - The issue stems from my erroneous translation of the labels
    - The way we figured this out is by feeding the model with correct but actually wrong answers, so that we can distinguish whether the issue lies in the content of the label or my code implementation
  - What I mean by wrong labels is that I use the previously well-tested synthetic radar dataset labels for training
![](https://i.imgur.com/H4HTexN.jpg)
  - It is trainable with correct but actually wrong labels
![](https://i.imgur.com/tKcw2LA.png)
  - When testing ```PASCAL_VOC``` dataset, I actually used padding for the input images, but I forgot that padding existed. So we can now confirm that my code can only take square inputs
  - Remove useless transforms of ```YOLOv3-VOC```
    - we need ```LongestMaxSize()``` and ```PadIfNeeded()``` to avoid ```RuntimeError: Trying to resize storage that is not resizable```
    - we need ```Normalize()``` to avoid ```RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.HalfTensor) should be the same```
    - we need ```ToTensorV2()``` to avoid ```RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[10, 416, 416, 3] to have 3 channels, but got 416 channels instead```
- 2023.03.14
  - Ref. Albumentations Documentation [Full API Reference](https://albumentations.ai/docs/api_reference/full_reference/)
    - testing different border modes
    ![](https://i.imgur.com/5m01r0U.png)
    - comparison of the 4 different modes: 
    ![](https://i.imgur.com/EOvisqk.png)
    - ```cv2.BORDER_CONSTANT```, ```cv2.BORDER_REFLECT```, ```cv2.BORDER_DEFAULT```, ```cv2.BORDER_REPLICATE``` with the value of ```0```, ```2```, ```4``` and ```1```, respectively
  - Remove useless transforms of ```YOLOv3-VOC```
    - we need ```LongestMaxSize()``` and ```PadIfNeeded()``` to avoid ```RuntimeError: Trying to resize storage that is not resizable```
    - we need ```Normalize()``` to avoid ```RuntimeError: Input type (torch.cuda.ByteTensor) and weight type (torch.cuda.HalfTensor) should be the same```
    - we need ```ToTensorV2()``` to avoid ```RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[10, 416, 416, 3] to have 3 channels, but got 416 channels instead```
  - The execution result and the error messages of the same code are different when using my PC compared to the lab PC, which is weird and annoying.
- 2023.03.10
  - Still untrainable
    - First, I prepare ```3``` types of square sizes of images, 64-by-64, 256-by-256, and 416-by-416, respectively.
    - The way I tested it is by simply changing the input images to the previously successful version, without changing anything else, and seeing how it goes.
    - Even though I resized all the images to a square size, the exact same error persists. Specifically:
      ```clike!
      RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[16, 64, 64, 3] to have 3 channels, but got 64 channels instead
      ```
      ```clike!
      RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[16, 256, 256, 3] to have 3 channels, but got 256 channels instead
      ```
      ```clike!
      RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[16, 416, 416, 3] to have 3 channels, but got 416 channels instead
      ```
  - It still doesn't work, but every piece of code is the same, so I speculate that maybe it's because the images are not actually encoded in the ```'JPEG'``` format.
  - So I re-read the dataset, stored the ```.mat``` files out, and converted the ```.mat``` files into scaled color and grayscale.
    - Plotting 7193 frames of the CARRADA Dataset in scaled color using MATLAB [link](https://www.youtube.com/watch?v=DyZ7rPXPHjE)
  - Then I used the scaled color images to train, still getting errors, but at least now we have a different error message.
    ```clike!
    ValueError: Expected x_min for bbox (-0.103515625, 0.306640625, 0.224609375, 0.365234375, 2.0) to be in the range [0.0, 1.0], got -0.103515625.
    ```
- 2023.03.09
  - The function for converting ```.mat``` files to ```.jpg``` images
    <img src=https://i.imgur.com/HLDGo78.png width=75% height=75%>
    <img src=https://i.imgur.com/XqgPJW8.png width=75% height=75%>
- 2023.03.04
  - New breach, image file format may be the issue
  - Regenerate all data in .jpg
- 2023.02.21
  - Modified from YOLO-CFAR
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
  - Modified from YOLO-Pascal_VOC
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


## Reference
- The original code was copied from [YOLOv3-PyTorch](https://github.com/SannaPersson/YOLOv3-PyTorch)
  
