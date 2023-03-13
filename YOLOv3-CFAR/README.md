# YOLOv3-stable
- Dependencies for env ```pt3.7```
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
  ```
- Dataset: ```RD_maps``` synthetic radar dataset
- Note
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

