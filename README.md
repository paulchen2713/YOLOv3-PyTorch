# YOLOv3-PyTorch
- This is the research topic for my master's thesis. You are welcome to play with the code, but please don't hijack my research. I'll die.
- A guide on 'how to use this code'
  - First, download the ```CARRADA Dataset```, the ```Pascal_VOC Dataset``` from kaggle, or the ```CFAR Dataset``` and structure the folders as shown in the file tree below
  - Second, set up a virtual environment using Anaconda, e.g. 
    - ```conda create --name pt3.7 python=3.7``` 
    - ```conda create --name pt3.8 python=3.8```
  - Before installing any packages, remember to enter your conda virtual environment, e.g. 
    - ```conda activate pt3.7```
    - ```conda activate pt3.8```
  - Third, you can manually install all the packages that you need, or you can install with ```pip install -r requirements.txt```
  - Then, copy the code to anywhere you like, and make sure you have changed the file path in ```config.py``` before running the code
    - Just click the 'run' button and see the results
  - Caveats:
    - There are a bunch of dead code, commented code, and outdated comments in my program
    - I use ```albumentations``` library solely for the purpose of padding
  - Dataset file tree 
    ```python
    D:\Datasets\RADA\RD_JPG>tree
    D:.
    ├─checks
    ├─images
    ├─imagesc
    ├─imwrite
    ├─labels
    ├─mats
    └─training_logs
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
  - Stable dependency
    - for python 3.7 
        ```python
        python==3.7.13
        numpy==1.19.2
        pytorch==1.7.1
        torchaudio==0.7.2
        torchvision==0.8.2
        pandas==1.2.1
        pillow==8.1.0 
        tqdm==4.56.0
        albumentations==0.5.2 
        matplotlib==3.3.4
        ```
    - for python 3.8
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
        albumentations==1.3.0
        matplotlib==3.6.2
        ```
    - It's well tested, and the code can be properly executed under these settings



## Notes
- 2023.05.01
  - The training duration is ```5.7350 hours``` with ```WEIGHT_DECAY = 1e-4``` and ```LEARNING_RATE = 10e-5```
    ```
    --------------------------------------------------
    The stats of 2023-05-01-1 training:
    --------------------------------------------------
    max mAP:  0.43861937522888184
    mean mAP: 0.3436750084161758

    max training loss: 140.25660705566406
    min training loss: 0.8655197024345398

    max training loss on average: 18.507166700363157
    min training loss on average: 1.1550286275148391

    min training accuracy: 4.963176250457764
    max training accuracy: 94.96363067626953

    min testing accuracy: 36.68720245361328
    max testing accuracy: 68.37782287597656
    --------------------------------------------------
    ```
  - The training duration is ```7.0366 hours``` with ```WEIGHT_DECAY = 1e-4``` and ```LEARNING_RATE = 11e-5```
    ```
    --------------------------------------------------
    The stats of 2023-05-01-2 training:
    --------------------------------------------------
    max mAP:  0.449009507894516
    mean mAP: 0.38735678046941757

    max training loss: 102.38961791992188
    min training loss: 0.9561270475387573

    max training loss on average: 17.273788038889567
    min training loss on average: 1.2045013213157654

    min training accuracy: 3.2843875885009766
    max training accuracy: 96.39083099365234

    min testing accuracy: 35.68332290649414
    max testing accuracy: 73.03217315673828
    --------------------------------------------------
    ```
  - The training duration is ```7.1689 hours``` with ```WEIGHT_DECAY = 1e-4``` and ```LEARNING_RATE = 12e-5```
    ```
    --------------------------------------------------
    The stats of 2023-05-01-3 training:
    --------------------------------------------------
    max mAP:  0.4372125566005707
    mean mAP: 0.38275537043809893

    max training loss: 125.8486099243164
    min training loss: 0.9757415056228638

    max training loss on average: 17.398162371317547
    min training loss on average: 1.2320519105593364

    min training accuracy: 1.1024198532104492
    max training accuracy: 94.62055206298828

    min testing accuracy: 34.86196517944336
    max testing accuracy: 73.3515853881836
    --------------------------------------------------
    ```
- 2023.04.30
  - The training duration is ```7.0542 hours``` with ```WEIGHT_DECAY = 1e-4``` and ```LEARNING_RATE = 6e-5```
    ```
    --------------------------------------------------
    The stats of 2023-04-29-1 training:
    --------------------------------------------------
    max mAP:  0.4267594814300537
    mean mAP: 0.3732090950012207

    max training loss: 70.09312438964844
    min training loss: 0.9483757019042969

    max training loss on average: 20.225014870961505
    min training loss on average: 1.1955717974901199

    min training accuracy: 0.8279584646224976
    max training accuracy: 95.35245513916016

    min testing accuracy: 32.3294563293457
    max testing accuracy: 73.48847961425781
    --------------------------------------------------
    ```
  - The training duration is ```7.1015 hours``` and ```WEIGHT_DECAY = 1e-4``` and ```LEARNING_RATE = 7e-5```
    ```
    --------------------------------------------------
    The stats of 2023-04-29-2 training:
    --------------------------------------------------
    max mAP:  0.4309697151184082
    mean mAP: 0.37477160841226576

    max training loss: 105.76203155517578
    min training loss: 0.8929504752159119

    max training loss on average: 20.704750878016153
    min training loss on average: 1.1069866104920705

    min training accuracy: 4.180961608886719
    max training accuracy: 96.9443359375

    min testing accuracy: 37.37166213989258
    max testing accuracy: 77.36710357666016
    --------------------------------------------------
    ```
  - The training duration is ```6.7780 hours``` with ```WEIGHT_DECAY = 1e-4``` and ```LEARNING_RATE = 8e-5```
    ```
    --------------------------------------------------
    The stats of 2023-04-30-1 training:
    --------------------------------------------------
    max mAP:  0.4340965747833252
    mean mAP: 0.36167612075805666

    max training loss: 104.89147186279297
    min training loss: 0.9307739734649658

    max training loss on average: 19.40190040588379
    min training loss on average: 1.1852473825216294

    min training accuracy: 1.6238964796066284
    max training accuracy: 95.42564392089844

    min testing accuracy: 30.458589553833008
    max testing accuracy: 71.52635192871094
    --------------------------------------------------
    ```
  - The training duration is ```5.5800 hours``` with ```WEIGHT_DECAY = 1e-4``` and ```LEARNING_RATE = 9e-5```
    ```
    --------------------------------------------------
    The stats of 2023-04-30-2 training:
    --------------------------------------------------
    max mAP:  0.43561217188835144
    mean mAP: 0.3712215393781662

    max training loss: 125.58506774902344
    min training loss: 0.9454944729804993

    max training loss on average: 18.3668585618337
    min training loss on average: 1.1890920907258988

    min training accuracy: 2.2048397064208984
    max training accuracy: 96.52349090576172

    min testing accuracy: 30.778005599975586
    max testing accuracy: 72.59867858886719
    --------------------------------------------------
    ```
- How to get model summary in PyTorch?
  - Using ```torchsummary``` to get the result
    ```python
    from torchsummary import summary
    # simple test settings
    num_classes = 3   # 
    num_examples = 20 # batch size
    num_channels = 3  # num_anchors

    model = YOLOv3(num_classes=num_classes) # initialize a YOLOv3 model as model

    # simple test with random inputs of 20 examples, 3 channels, and IMAGE_SIZE-by-IMAGE_SIZE input
    x = torch.randn((num_examples, num_channels, IMAGE_SIZE, IMAGE_SIZE))

    out = model(x) 

    # print out the model summary using third-party library called 'torchsummary'
    summary(model.cuda(), (3, 416, 416), bs=16)
    ```
    - model parameter summary
        ```cpp
        ================================================================
        Total params: 61,534,504
        Trainable params: 61,534,504
        Non-trainable params: 0
        ----------------------------------------------------------------
        Input size (MB): 31.69
        Forward/backward pass size (MB): 13175.06
        Params size (MB): 234.74
        Estimated Total Size (MB): 13441.48
        ----------------------------------------------------------------
        ```
  - Reference
    - stackoverflow [How do I print the model summary in ```PyTorch```?](https://stackoverflow.com/questions/42480111/how-do-i-print-the-model-summary-in-pytorch)
    - PyTorch Doc [Is there similar pytorch function as ```model.summary()``` as ```keras```?](https://discuss.pytorch.org/t/is-there-similar-pytorch-function-as-model-summary-as-keras/2678)
- 2023.04.29
  - The comparison between different ```WEIGHT_DECAY``` under the same ```LEARNING_RATE = 3e-5```
    - The ```loss``` value for every updates
      ![](https://i.imgur.com/c3i8F03.png)
    - The ```train-object-accuracy``` for every epochs
      ![](https://i.imgur.com/5oO2CyN.png)
    - The ```test-object-accuracy``` for every ```10``` epochs
      ![](https://i.imgur.com/XNy01W4.png)
    - The ```mAP``` for every ```10``` epochs
        ```cpp
        2023-04-27, epoch: 100, duration: 7.1676 hours, WEIGHT_DECAY = 1e-1, LEARNING_RATE = 3e-5, max mAP: 0.3289
        2023-04-26, epoch: 100, duration: 7.7900 hours, WEIGHT_DECAY = 1e-2, LEARNING_RATE = 3e-5, max mAP: 0.3646
        2023-04-25, epoch: 100, duration: 6.2753 hours, WEIGHT_DECAY = 1e-3, LEARNING_RATE = 3e-5, max mAP: 0.3603
        2023-04-22, epoch: 100, duration: 7.2117 hours, WEIGHT_DECAY = 1e-4, LEARNING_RATE = 3e-5, max mAP: 0.3792
        ```
      ![](https://i.imgur.com/eftQ9Tb.png)
- 2023.04.28
  - The training duration is ```7.5511 hours``` with ```WEIGHT_DECAY = 1e-4``` and ```LEARNING_RATE = 1e-5```
    ```
    --------------------------------------------------
    The stats of 2023-04-28 training:
    --------------------------------------------------
    max mAP:  0.2697495222091675
    mean mAP: 0.18186791352927684

    max training loss: 92.10067749023438
    min training loss: 1.1181566715240479

    max training loss on average: 32.7851714070638
    min training loss on average: 1.3748071026802062

    min training accuracy: 2.0996294021606445
    max training accuracy: 92.99666595458984

    min testing accuracy: 19.9406795501709
    max testing accuracy: 64.90988159179688
    --------------------------------------------------
    ```
  - The training duration is ```7.2838 hours``` with ```WEIGHT_DECAY = 1e-4``` and ```LEARNING_RATE = 2e-5```
    ```
    --------------------------------------------------
    The stats of 2023-04-27-2 training:
    --------------------------------------------------
    max mAP:  0.3233576714992523
    mean mAP: 0.23364422097802162

    max training loss: 67.91127014160156
    min training loss: 0.9422303438186646

    max training loss on average: 27.790054613749188
    min training loss on average: 1.224434497753779

    min training accuracy: 0.37967154383659363
    max training accuracy: 95.5811767578125

    min testing accuracy: 22.19940757751465
    max testing accuracy: 69.38169860839844
    --------------------------------------------------
    ```
  - The training duration is ```7.2117 hours``` with ```WEIGHT_DECAY = 1e-4``` and ```LEARNING_RATE = 3e-5```
    ```
    --------------------------------------------------
    The stats of 2023-04-22 training:
    --------------------------------------------------
    max mAP:  0.37920689582824707
    mean mAP: 0.3020245939493179

    max training loss: 72.82600402832031
    min training loss: 0.8917444944381714

    max training loss on average: 25.31787603378296
    min training loss on average: 1.1737037108341852

    min training accuracy: 0.5489227175712585
    max training accuracy: 96.67901611328125

    min testing accuracy: 28.838693618774414
    max testing accuracy: 70.72781372070312
    --------------------------------------------------
    ```
  - The training duration is ```8.1383 hours``` with ```WEIGHT_DECAY = 1e-4``` and ```LEARNING_RATE = 4e-5```
    ```
    --------------------------------------------------
    The stats of 2023-04-28-2 training:
    --------------------------------------------------
    max mAP:  0.3963651657104492
    mean mAP: 0.3341544926166534

    max training loss: 67.77149963378906
    min training loss: 0.9209076166152954

    max training loss on average: 22.19623363494873
    min training loss on average: 1.146754193107287

    min training accuracy: 0.7410457134246826
    max training accuracy: 96.36795806884766

    min testing accuracy: 33.926536560058594
    max testing accuracy: 70.9787826538086
    --------------------------------------------------
    ```
  - The training duration is ```7.1785 hours``` with ```WEIGHT_DECAY = 1e-4``` and ```LEARNING_RATE = 5e-5```
    ```
    --------------------------------------------------
    The stats of 2023-04-28-3 training:
    --------------------------------------------------
    max mAP:  0.41434526443481445
    mean mAP: 0.3443592220544815

    max training loss: 65.99470520019531
    min training loss: 0.8718012571334839

    max training loss on average: 20.811571718851724
    min training loss on average: 1.0752028383811314

    min training accuracy: 1.0612506866455078
    max training accuracy: 96.4731674194336

    min testing accuracy: 35.318275451660156
    max testing accuracy: 74.19575500488281
    --------------------------------------------------
    ```
- 2023.04.27
  - Performing a grid search to find the optimal weight decay setting, all tests have the same settings except for the weight decay parameter
    ![](https://i.imgur.com/iUf6L9i.png)
  - The training duration is ```7.1676 hours``` with  ```WEIGHT_DECAY = 1e-1```
    ```
    --------------------------------------------------
    The stats of 2023-04-27 training: 
    --------------------------------------------------
    max mAP:  0.328988641500473
    mean mAP: 0.26167472153902055

    max training loss: 57.49835968017578
    min training loss: 2.0004570484161377

    max training loss on average: 23.69808032989502
    min training loss on average: 2.260551511446635

    min training accuracy: 1.3860299587249756
    max training accuracy: 78.02479553222656

    min testing accuracy: 25.644535064697266
    max testing accuracy: 53.70750427246094
    --------------------------------------------------
    ```
    - mean mAP: ```0.26167472153902055```
        ![](https://i.imgur.com/Sp7dtJZ.png)
    - loss range: ```[57.49835968017578, 2.0004570484161377]```
    ![](https://i.imgur.com/nudeE2B.png)
    - max training accuracy: ```78.02479553222656```
    ![](https://i.imgur.com/LeC7FqK.png)
    - max testing accuracy: ```53.70750427246094```
    ![](https://i.imgur.com/q5ghBLQ.png)
  - The training duration is ```7.7900 hours``` with  ```WEIGHT_DECAY = 1e-2```
    ```
    --------------------------------------------------
    The stats of 2023-04-26 training: 
    --------------------------------------------------
    max mAP:  0.36460867524147034
    mean mAP: 0.2820669665932655

    max training loss: 59.46959686279297
    min training loss: 1.341576099395752

    max training loss on average: 24.546935682296752
    min training loss on average: 1.5733012755711873

    min training accuracy: 0.7913635969161987
    max training accuracy: 89.63908386230469

    min testing accuracy: 29.956649780273438
    max testing accuracy: 64.81861114501953
    --------------------------------------------------
    ```
    - mean mAP: ```0.2820669665932655```
    ![](https://i.imgur.com/au8uc9m.png)
    - loss range: ```[59.46959686279297, 1.341576099395752]```
    ![](https://i.imgur.com/dQZQtky.png)
    - max training accuracy: ```89.63908386230469```
    ![](https://i.imgur.com/Y5d47HL.png)
    - max testing accuracy: ```64.81861114501953```
    ![](https://i.imgur.com/t1zuJTp.png)
  - The training duration is ```6.2753 hours``` with ```WEIGHT_DECAY = 1e-3``` 
    ```
    --------------------------------------------------
    The stats of 2023-04-25 training: 
    --------------------------------------------------
    max mAP:  0.3603482246398926
    mean mAP: 0.2835115119814873

    max training loss: 61.669921875
    min training loss: 0.9460040330886841

    max training loss on average: 23.978200359344484
    min training loss on average: 1.233974441687266

    min training accuracy: 1.289968490600586
    max training accuracy: 95.745849609375

    min testing accuracy: 23.180469512939453
    max testing accuracy: 69.15354919433594
    --------------------------------------------------
    ```
    - mean mAP: ```0.2835115119814873```
    ![](https://i.imgur.com/5dtS7Sy.png)
    - loss range: ```[61.669921875, 0.9460040330886841]```
    ![](https://i.imgur.com/AQ5wZ7L.png)
    - max training accuracy: ```95.745849609375```
    ![](https://i.imgur.com/YUQTLki.png)
    - max testing accuracy: ```69.15354919433594```
    ![](https://i.imgur.com/ih4zhbz.png)
  - The training duration is ```7.2117``` hours with ```WEIGHT_DECAY = 1e-4``` 
    ```
    --------------------------------------------------
    The stats of 2023-04-22 training: 
    --------------------------------------------------
    max mAP:  0.37920689582824707
    mean mAP: 0.3020245939493179

    max training loss: 72.82600402832031
    min training loss: 0.8917444944381714

    max training loss on average: 25.31787603378296
    min training loss on average: 1.1737037108341852

    min training accuracy: 0.5489227175712585
    max training accuracy: 96.67901611328125

    min testing accuracy: 28.838693618774414
    max testing accuracy: 70.72781372070312
    --------------------------------------------------
    ```
    - mean mAP: ```0.3020245939493179```
    ![](https://i.imgur.com/YMORvXt.png)
    - loss range: ```[72.82600402832031, 0.8917444944381714]```
    ![](https://i.imgur.com/Zn8u5Dy.png)
    - max training accuracy: ```96.67901611328125```
    ![](https://i.imgur.com/3kdOs6a.png)
    - max testing accuracy: ```70.72781372070312```
    ![](https://i.imgur.com/vKFuAXX.png)
- 2023.04.26
  - Performing a grid search to find the optimal weight decay setting, all tests have the same settings except for the weight decay parameter
    ![](https://i.imgur.com/iUf6L9i.png)
  - The training duration is ```6.2753 hours``` with  ```WEIGHT_DECAY = 1e-3``` 
    ```
    --------------------------------------------------
    The stats of 2023-04-25 training: 
    --------------------------------------------------
    max mAP:  0.3603482246398926
    mean mAP: 0.2835115119814873

    max training loss: 61.669921875
    min training loss: 0.9460040330886841

    max training loss on average: 23.978200359344484
    min training loss on average: 1.233974441687266

    min training accuracy: 1.289968490600586
    max training accuracy: 95.745849609375

    min testing accuracy: 23.180469512939453
    max testing accuracy: 69.15354919433594
    --------------------------------------------------
    ```
  - The training duration is ```7.7900 hours``` with  ```WEIGHT_DECAY = 1e-2```
    ```
    --------------------------------------------------
    The stats of 2023-04-26 training: 
    --------------------------------------------------
    max mAP:  0.36460867524147034
    mean mAP: 0.2820669665932655

    max training loss: 59.46959686279297
    min training loss: 1.341576099395752

    max training loss on average: 24.546935682296752
    min training loss on average: 1.5733012755711873

    min training accuracy: 0.7913635969161987
    max training accuracy: 89.63908386230469

    min testing accuracy: 29.956649780273438
    max testing accuracy: 64.81861114501953
    --------------------------------------------------
    ```
- 2023.04.24
  - The train and test settings
    ![](https://i.imgur.com/6FZ7c9e.png)
- The result of training for ```100``` epochs, with ```k_means()``` anchor that rounded to ```3``` decimal places 
  - The training duration: ```7.2117``` hours
    ```
    --------------------------------------------------
    The stats of 2023-04-22 training: 
    --------------------------------------------------
    max mAP:  0.37920689582824707
    mean mAP: 0.3020245939493179

    max training loss: 72.82600402832031
    min training loss: 0.8917444944381714

    max training loss on average: 25.31787603378296
    min training loss on average: 1.1737037108341852

    min training accuracy: 0.5489227175712585
    max training accuracy: 96.67901611328125

    min testing accuracy: 28.838693618774414
    max testing accuracy: 70.72781372070312
    --------------------------------------------------
    ```
- The result of training for ```300``` epochs, with same anchors above
  - The training duration: ```20.8263``` hours
    ```
    --------------------------------------------------
    The stats of 2023-04-23 training: 
    --------------------------------------------------
    max mAP:  0.4179251194000244
    mean mAP: 0.3632150818904241

    max training loss: 72.01780700683594
    min training loss: 0.5801995992660522

    max training loss on average: 24.274858560562134
    min training loss on average: 0.7920041881004969

    min training accuracy: 0.45743560791015625
    max training accuracy: 99.13544464111328

    min testing accuracy: 35.75177001953125
    max testing accuracy: 72.34770965576172
    --------------------------------------------------
    ```
- The figures for the stats
  - max mAP:  ```0.4179251194000244```
    ![](https://i.imgur.com/aB3nZLB.png)
  - loss range: ```[72.82600402832031, 0.8917444944381714]```
    ![](https://i.imgur.com/qTgtTPY.png)
  - max training accuracy: ```99.13544464111328```
    ![](https://i.imgur.com/kODi2F4.png)
  - max testing accuracy: ```72.34770965576172```
    ![](https://i.imgur.com/bWHb0MP.png)
- 2023.04.23
  - Script for plotting the figures ```plot_training_state.py```
    ![](https://i.imgur.com/lRGOLFg.png)
    ![](https://i.imgur.com/VXSmBJJ.png)
    ![](https://i.imgur.com/rTh7e1x.png)
    ![](https://i.imgur.com/EOHHXsY.png)
    ![](https://i.imgur.com/CDqXE74.png)
    ![](https://i.imgur.com/R08ByyL.png)
    ![](https://i.imgur.com/CQO3qML.png)
- 2023.04.21
  - Script for creating random samples ```create_csv.py```
    ![](https://i.imgur.com/FGOJ31Z.png)
    ![](https://i.imgur.com/aIZdzWG.png)
- 2023.04.18
  - The third clustering result using custom ```k_means()```
    ```python
    Number of clusters: 9
    Average IoU: 0.6639814720619468

    Anchors original: 
    (0.42412935323383083, 0.09495491293532338), (0.040049518201284794, 0.04793729925053533), (0.12121121241202815, 0.02474208253358925), 
    (0.21935948581560283, 0.041091810726950354), (0.015625, 0.016347497459349592), (0.21888516435986158, 0.09671009948096886), 
    (0.038657583841463415, 0.008815858422256097), (0.125454418344519, 0.07256711409395973), (0.058373810467882634, 0.018722739888977002), 

    Anchors rounded to 2 decimal places: 
    (0.42, 0.09), (0.04, 0.05), (0.12, 0.02), 
    (0.22, 0.04), (0.02, 0.02), (0.22, 0.10), 
    (0.04, 0.01), (0.13, 0.07), (0.06, 0.02), 

    Anchors rounded to 3 decimal places: 
    (0.424, 0.095), (0.040, 0.048), (0.121, 0.025), 
    (0.219, 0.041), (0.016, 0.016), (0.219, 0.097), 
    (0.039, 0.009), (0.125, 0.073), (0.058, 0.019), 
    ```
  - The comparison of 
    - original anchor for general image dataset 
    ```python
    (0.28, 0.22), (0.38, 0.48), (0.9,  0.78),
    (0.07, 0.15), (0.15, 0.11), (0.14, 0.29),
    (0.02, 0.03), (0.04, 0.07), (0.08, 0.06)
    ```
    - ```sklearn.cluster.KMeans()``` result
    ```python
    (0.211, 0.098), (0.339, 0.087), (0.495, 0.092),
    (0.158, 0.033), (0.232, 0.043), (0.125, 0.082), 
    (0.033, 0.017), (0.065, 0.027), (0.107, 0.024),
    ```
    - ```sklearn.cluster.MiniBatchKMeans()``` result
    ```python
    (0.329, 0.085), (0.424, 0.096), (0.530, 0.089),
    (0.157, 0.031), (0.232, 0.064), (0.164, 0.094),
    (0.027, 0.016), (0.056, 0.024), (0.105, 0.029),
    ```
    - Custom ```k_means()``` result
    ```python
    (0.125, 0.073), (0.219, 0.097), (0.424, 0.095),
    (0.040, 0.048), (0.121, 0.025), (0.219, 0.041),
    (0.016, 0.016), (0.039, 0.009), (0.058, 0.019),
    ```
  - training for 1000 epochs with original anchors
    ```
    max mAP:  0.18192845582962036 (the highest mAP obtained out of 10 tests)
    mean mAP: 0.1663009986281395  (the average mAP obtained out of 10 tests)
    
    max training loss: 125.03005981445312
    min training loss: 0.6005923748016357
    
    max training loss on average: 19.55863230228424
    min training loss on average: 0.8333272246519724
    
    min training accuracy: 2.8318750858306885
    max training accuracy: 98.84278869628906

    min testing accuracy: 33.172786712646484
    max testing accuracy: 70.57997131347656
    ```
    - The figures fot the stats
    ![](https://i.imgur.com/XzHCdeK.png)
    ![](https://i.imgur.com/4uwzdv8.png)
    ![](https://i.imgur.com/n1YbQBr.png)
    ![](https://i.imgur.com/T6eq1gA.png)
    ![](https://i.imgur.com/w1rvagY.png)
  - training for 100 epochs with ```sklearn.cluster.KMeans()``` anchor that rounded to 2 decimal places
    ```
    max training loss on average: 17.887332406044006
    min training loss on average: 1.1761843407154082
    
    min training accuracy: 1.1478031873703003
    max training accuracy: 96.33079528808594

    min testing accuracy: 28.48825454711914
    max testing accuracy: 67.01465606689453
    
    max mAP:  0.1628512293100357
    mean mAP: 0.1628512293100357 (only test once)
    ```
  - training for 100 epochs with ```sklearn.cluster.KMeans()``` anchor that rounded to 3 decimal places
    ```
    max training loss on average: 18.193040917714438
    min training loss on average: 1.2186308292547863
    
    min training accuracy: 4.069056510925293
    max training accuracy: 94.63731384277344

    min testing accuracy: 28.80947494506836
    max testing accuracy: 66.93435668945312
    
    max mAP:  0.17361223697662354
    mean mAP: 0.17361223697662354 (only test once)
    ```
- The YOLO network seems not able to properly learn this task
  - Keep improving the anchor settings
    - Plot the comparison between different anchor settings
  - Redesign the feture extractor structure
    - Change the detection head network
  - Apply certain training strategy for our task, e.g. Weight Initialization:
    - Random Initialization (current method)
    - Xavier Initialization, or Glorot Initialization
    - Kaiming Initialization, or He Initialization
    - LeCun Initialization
    - Ref. Deeplizard [Weight Initialization Explained](https://deeplizard.com/learn/video/8krd5qKVw-Q)
  - Using k-fold cross-validation to ensure that there's no training data selection bias
- 2023.04.17
  - The code for handcrafted-from-scratch version of ```k_means()``` which consider IoU in its distance metric
    ![](https://i.imgur.com/L4O0tu4.png)
    ![](https://i.imgur.com/i730QvA.png)
  - The first clustering result using ```sklearn.cluster.KMeans()``` 
    ```python
    Estimator: KMeans(n_clusters=9, verbose=True)
    Number of Clusters: 9
    Average IoU: 0.6268763251152744
    Inertia: 4.175114625246291
    Silhouette Score: 0.4465142389008657
    Date and Duration: 2023-04-13 / 0.0951 seconds

    Anchors: 
      1: (0.03258875446251471852, 0.01661357100357002681)   5.414155861808978
      2: (0.06474560301507539806, 0.02702967964824120467)   17.50052908129688
      3: (0.10668965880370681609, 0.02383240311710192738)   25.426709570360032
      4: (0.15826612903225806273, 0.03252153592375366803)   51.47057600836014
      5: (0.23229679802955666146, 0.04291102216748768350)   99.68093049682716
      6: (0.12471330275229357276, 0.08154147553516821745)   101.69306725286172
      7: (0.21058315334773208827, 0.09842400107991366998)   207.26436512508812
      8: (0.33944144518272417743, 0.08742992109634553644)   296.77338769155074
      9: (0.49540441176470573215, 0.09187346813725494332)   455.1452143932022

    Anchors original: 
    (0.03258875446251472, 0.016613571003570027), (0.0647456030150754, 0.027029679648241205), (0.10668965880370682, 0.023832403117101927), 
    (0.15826612903225806, 0.03252153592375367), (0.23229679802955666, 0.042911022167487683), (0.12471330275229357, 0.08154147553516822), 
    (0.2105831533477321, 0.09842400107991367), (0.3394414451827242, 0.08742992109634554), (0.49540441176470573, 0.09187346813725494), 

    Anchors rounded to 2 decimal places: 
    (0.03, 0.02), (0.06, 0.03), (0.11, 0.02), 
    (0.16, 0.03), (0.23, 0.04), (0.12, 0.08), 
    (0.21, 0.10), (0.34, 0.09), (0.50, 0.09), 

    Anchors rounded to 3 decimal places: 
    (0.033, 0.017), (0.065, 0.027), (0.107, 0.024), 
    (0.158, 0.033), (0.232, 0.043), (0.125, 0.082), 
    (0.211, 0.098), (0.339, 0.087), (0.495, 0.092), 
    ```
  - The second clustering result using
    ```python
    Estimator: MiniBatchKMeans(n_clusters=9, tol=0.0001, verbose=True)
    Number of Clusters: 9
    Average IoU: 0.6075905487924542
    Inertia: 4.375712040766109
    Silhouette Score: 0.41462042329969084
    Date and Duration: 2023-04-13 / 0.0423 seconds

    Anchors: 
      1: (0.02677950180907319802, 0.01550867137489563008)   4.153144931403392
      2: (0.05614595190665907370, 0.02351197887023335348)   13.201024348785062
      3: (0.10527306967984934039, 0.02908427495291902171)   30.61790903706541
      4: (0.15678998161764706731, 0.03086224724264705413)   48.388911778539104
      5: (0.23159116755117511999, 0.06435983699772555855)   149.0516979370658
      6: (0.16395052370452040114, 0.09384044239250277641)   153.85189674914707
      7: (0.32857417864476384795, 0.08490278490759754770)   278.9686281566692
      8: (0.42449951171874988898, 0.09640502929687500000)   409.23887863755215
      9: (0.53048469387755103899, 0.08938137755102043558)   474.1545270850689

    Anchors original: 
    (0.026779501809073198, 0.01550867137489563), (0.056145951906659074, 0.023511978870233353), (0.10527306967984934, 0.02908427495291902), 
    (0.15678998161764707, 0.030862247242647054), (0.23159116755117512, 0.06435983699772556), (0.1639505237045204, 0.09384044239250278), 
    (0.32857417864476385, 0.08490278490759755), (0.4244995117187499, 0.096405029296875), (0.530484693877551, 0.08938137755102044), 

    Anchors rounded to 2 decimal places: 
    (0.03, 0.02), (0.06, 0.02), (0.11, 0.03), 
    (0.16, 0.03), (0.23, 0.06), (0.16, 0.09), 
    (0.33, 0.08), (0.42, 0.10), (0.53, 0.09), 

    Anchors rounded to 3 decimal places: 
    (0.027, 0.016), (0.056, 0.024), (0.105, 0.029), 
    (0.157, 0.031), (0.232, 0.064), (0.164, 0.094), 
    (0.329, 0.085), (0.424, 0.096), (0.530, 0.089),
    ```
  - The original anchor for general image dataset
    ```python
    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9,  0.78)], 
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], 
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], 
    ]  # Note these have been rescaled to be between [0, 1]
    ```
- 2023.04.13
  - stackoverflow [Custom Python list sorting](https://stackoverflow.com/questions/11850425/custom-python-list-sorting)
    ```python
    from functools import cmp_to_key
    cmp_key = cmp_to_key(cmp_function)
    mylist.sort(key=cmp_key)
    ```
  - ```get_anchors2.py```
    - Finishing the part where I use  ```sklearn.cluster.KMeans()``` and ```sklearn.cluster.MiniBatchKMeans()``` for clustering
    - The custom-designed / handcrafted-from-scratch version of ```k_means()``` is also finished, but it hasn't been well-tested yet
  - The part of the code
    ![](https://i.imgur.com/2S6N6t9.png)
    ![](https://i.imgur.com/254IZGY.png)
    ![](https://i.imgur.com/FLtmwaM.png)
    ![](https://i.imgur.com/fUJmBSC.png)
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
- The implementation is based on the following paper
  - Redmon, Joseph, and Ali Farhadi. "[Yolov3: An incremental improvement](https://arxiv.org/abs/1804.02767)." *arXiv preprint arXiv:1804.02767* (2018).
    ```
      @article{yolov3,
      title={YOLOv3: An Incremental Improvement},
      author={Redmon, Joseph and Farhadi, Ali},
      journal = {arXiv},
      year={2018}
    }
    ```
- The original code was copied from [YOLOv3-PyTorch](https://github.com/SannaPersson/YOLOv3-PyTorch) and for more details please read their Medium post [YOLOv3 — Implementation with Training setup from Scratch](https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0)
  
  
