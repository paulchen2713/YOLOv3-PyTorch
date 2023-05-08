# YOLOv3-PyTorch-model-summary
- Using ```torchsummary.summary()``` to get the result
  - sample code
    ```python
    # from torchsummary import summary  # changing the way of import due to naming conflicts
    import torchsummary  
    
    # simple test settings
    IMAGE_SIZE = 416  # multiples of 32 are workable with stride [32, 16, 8]
    num_classes = 3   # 
    batch_size = 20   # num_examples
    num_channels = 3  # num_anchors

    model = YOLOv3(num_classes=num_classes) # initialize a YOLOv3 model as model

    # simple test with random inputs of 20 examples, 3 channels, and IMAGE_SIZE-by-IMAGE_SIZE input
    x = torch.randn((batch_size, num_channels, IMAGE_SIZE, IMAGE_SIZE))

    out = model(x) 

    # print out the model summary using third-party library called 'torchsummary'
    torchsummary.summary(model.cuda(), (num_channels, IMAGE_SIZE, IMAGE_SIZE), bs=batch_size)
    ```
  - model parameter summary
    ```clike
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
- Using ```torchinfo.summary()``` to get the result
  - sample code
    ```python
    import torchsummary            # torchsummary.summary()
    from torchinfo import summary  # torchinfo.summary()

    # simple test settings
    IMAGE_SIZE = 416  # multiples of 32 are workable with stride [32, 16, 8]
    num_classes = 3   # 
    batch_size = 20   # num_examples
    num_channels = 3  # num_anchors

    model = YOLOv3(num_classes=num_classes) # initialize a YOLOv3 model as model

    # simple test with random inputs of 20 examples, 3 channels, and IMAGE_SIZE-by-IMAGE_SIZE input
    x = torch.randn((batch_size, num_channels, IMAGE_SIZE, IMAGE_SIZE))

    out = model(x)

    # print out the model summary using torchinfo.summary()
    summary(model.cuda(), input_size=(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE))
    ```
  - model parameter summary
    ```clike
    ====================================================================================================
    Total params: 61,534,648
    Trainable params: 61,534,648
    Non-trainable params: 0
    Total mult-adds (G): 653.05
    ====================================================================================================
    Input size (MB): 41.53
    Forward/backward pass size (MB): 12265.99
    Params size (MB): 246.14
    Estimated Total Size (MB): 12553.66
    ====================================================================================================
    ```

