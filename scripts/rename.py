
import torchvision.transforms as T # for resizing the images
from PIL import Image              # for loading and saving the images
import os                          # for grabbing the files

isResized = False
date = '2023.03.08'
PATH = f"C:/Users/paulc/Downloads/rename_files/"


def rename_files():
    count_jpg, count_png = 1, 1

    src_path = PATH + f"source/"
    dst_path = PATH + f"output/"

    for images in os.listdir(src_path):
        # check if the image ends with '.jpg' or '.jpeg'
        if images.endswith(f".jpg") or images.endswith(f".jpeg"):
            # print(count_jpg, src_path + images)

            # read the input image
            img = Image.open(src_path + images)

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
            img = img.save(dst_path + f'{date}-{count_jpg}.jpg')
            
            print(count_jpg)
            count_jpg += 1
    
    for images in os.listdir(src_path):
        # check if the image ends with '.png'
        if images.endswith(f".png"):
            # print(count_png, src_path + images)

            # read the input image
            img = Image.open(src_path + images)

            # overwrite the original image with the resized one
            img = img.save(dst_path + f'{date}-{count_jpg}.png')

            print(count_png)
            count_png += 1



if __name__ == '__main__':
    rename_files()


