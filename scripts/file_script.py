import os

# set the dataset path
DATASET = 'D:/Datasets/CARRADA/'

# directory names, number of directorie: 30
dir_names = ['2019-09-16-12-52-12', '2019-09-16-12-55-51', '2019-09-16-12-58-42', '2019-09-16-13-03-38', '2019-09-16-13-06-41', 
             '2019-09-16-13-11-12', '2019-09-16-13-13-01', '2019-09-16-13-14-29', '2019-09-16-13-18-33', '2019-09-16-13-20-20', 
             '2019-09-16-13-23-22', '2019-09-16-13-25-35', '2020-02-28-12-12-16', '2020-02-28-12-13-54', '2020-02-28-12-16-05', 
             '2020-02-28-12-17-57', '2020-02-28-12-20-22', '2020-02-28-12-22-05', '2020-02-28-12-23-30', '2020-02-28-13-05-44', 
             '2020-02-28-13-06-53', '2020-02-28-13-07-38', '2020-02-28-13-08-51', '2020-02-28-13-09-58', '2020-02-28-13-10-51', 
             '2020-02-28-13-11-45', '2020-02-28-13-12-42', '2020-02-28-13-13-43', '2020-02-28-13-14-35', '2020-02-28-13-15-36']

# number of images / labels in each directory, total number of labels: 7193
num_of_images = [286, 273, 304, 327, 218, 219, 150, 208, 152, 174, 
                 174, 235, 442, 493, 656, 523, 350, 340, 304, 108, 
                 129, 137, 171, 143, 104, 81, 149, 124, 121, 98]

# e.g. read "validated_seqs.txt"
def read_txt_file(file_name=""):
    dir_names = list()
    with open(DATASET + file_name, "r") as seqs_file:
        dir_names = seqs_file.read().splitlines()
    return dir_names


def create_folders():
    for dir_name in dir_names: # [23:24]:
        # e.g. "D:/Datasets/CARRADA/2020-02-28-13-09-58/annotations/box/"
        print(f"current directory: {dir_name}")

        directory = "images" # "labels", "RD_maps"

        parent_dir = DATASET + f"{dir_name}/" + "RD_maps"

        path = os.path.join(parent_dir, directory)

        # create foler named "path", e.g. '2020-02-28-13-09-58'
        # os.makedirs(path)

