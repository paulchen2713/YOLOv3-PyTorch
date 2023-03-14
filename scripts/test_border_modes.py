# Python program to explain cv2.copyMakeBorder() method 

import cv2 

# I want a visual example for the following border modes
modes = [cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT, cv2.BORDER_DEFAULT, cv2.BORDER_REPLICATE]
names = ['cv2.BORDER_CONSTANT', 'cv2.BORDER_REFLECT', 'cv2.BORDER_DEFAULT', 'cv2.BORDER_REPLICATE']

image_index = 98 # range from 97 to 100

for i in range(0, 4):
    # Set the file path 
    path = f'C:/Users/Paul/Downloads/YOLOv3-PyTorch-main/scripts/images/{image_index}.jpg'

    # Reading an image in default mode
    image = cv2.imread(path)

    # Window name in which image is displayed
    window_name = f'{names[i]}'

    # Using cv2.copyMakeBorder() method
    print(f"constant value: {modes[i]}")
    image = cv2.copyMakeBorder(image, 10 , 10 , 10 , 10 , modes[i])

    # Displaying the image 
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

