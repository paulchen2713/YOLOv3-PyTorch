
import torchvision.transforms as T # for resizing the images
from PIL import Image              # for loading and saving the images
import os
from os import listdir

isResized = False
date = '2022.06.22'
seq_path = 'D:/BeginnerPythonProjects/renaming_files/input/'

def rename_files():
    count = 1
    for images in os.listdir(seq_path):
        # check if the image ends with png
        if (images.endswith(f".jpg") or images.endswith(f".png")):
            # print(count, seq_path + images)

            # read the input image
            img = Image.open(seq_path + images)

            if isResized == True:
                print("Resizing and renaming images!")
                # # compute the size (width, height) of image
                # before = img.size
                # print(f"original image size: {before}")

                # define the transform function to resize the image with given size
                transform = T.Resize(size=(256, 64))

                # apply the transform on the input image
                img = transform(img)

                # # check the size (width, height) of image
                # after = img.size
                # print(f"resized image size: {after}")

            # overwrite the original image with the resized one
            img = img.save(f'D:/BeginnerPythonProjects/renaming_files/output/{date}-0{count}.png')
            
            print(count)
            count += 1



if __name__ == '__main__':
    rename_files()


