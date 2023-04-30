# YOLOv3-PyTorch-model-summary
- Using ```torchsummary``` to get the result
  - sample code
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
        ```
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


