# -*- coding: utf-8 -*-
"""
Created on Wen Apr 12 16:58:30 2023

@patch: 
    2023.04.12
    2023.04.23
    2023.04.28

@author: Paul
@file: plot_training_state.py
@dependencies:
    env pt3.8
    python==3.8.16
    matplotlib==3.6.2

Plot the log messages
"""

# from matplotlib import pyplot as plt  # I don't know which way of import is better
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import date
import os

# the path where we placed the log files
LOG_PATH = f"D:/Datasets/RADA/RD_JPG/training_logs/"
FIG_PATH = f"D:/Datasets/RADA/RD_JPG/stats_figures/"

folders = ['mAP', 'train', 'test']
data_folders = [
    'losses', 
    'mean_loss', 
    'class_accuracy', 
    'no_object_accuracy', 
    'object_accuracy'
]

logs = [
    '2023-04-07',
    '2023-04-15',
    '2023-04-16',
    '2023-04-22',
    '2023-04-23',
    '2023-04-25',
    '2023-04-26',
    '2023-04-27',
    '2023-04-27-2',
    '2023-04-28',
    '2023-04-28-2',
    '2023-04-28-3',
]
log_index = 8

# make sure we are using valid list subscripts
assert log_index <= len(logs)

# the file tree structure for the training logs:
"""
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
"""

# the file tree structure for the plots:
"""
D:\Datasets\RADA\RD_JPG\stats_figures>tree
D:.
├─2023-03-16
├─2023-03-27
├─2023-04-07
├─2023-04-15
├─2023-04-16
├─2023-04-22
├─2023-04-23
├─2023-04-25
├─2023-04-26
├─2023-04-27-1
├─2023-04-27-2
├─2023-04-28-1
└─2023-04-28-2
"""


def get_file_content(file_name):
    with open(file_name) as f:
        return [line.strip() for line in f]


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as text_file:
        # print(f"current file: {file_path}")
        data = []
        for line in text_file:
            data.append(float(line))
    return data


# read the training and testing statistics results
curr_txt_file = logs[log_index]  # get the current file name

mAP = load_data(LOG_PATH + f"mAP/{curr_txt_file}.txt")                                    # store mean Average Precision

# store the actual loss for every batch, the total number would be 'epoch x split', 
# split = num_train_samples=6000 / batch_size=20 = 300
losses = load_data(LOG_PATH + f"train/losses/{curr_txt_file}.txt") 
# store the average loss for every epoch
mean_loss = load_data(LOG_PATH + f"train/mean_loss/{curr_txt_file}.txt")

train_class_acc = load_data(LOG_PATH + f"train/class_accuracy/{curr_txt_file}.txt")       # store train class accuracy
train_no_obj_acc = load_data(LOG_PATH + f"train/no_object_accuracy/{curr_txt_file}.txt")  # store train no object accuracy
train_obj_acc = load_data(LOG_PATH + f"train/object_accuracy/{curr_txt_file}.txt")        # store train object accuracy

test_class_acc = load_data(LOG_PATH + f"test/class_accuracy/{curr_txt_file}.txt")         # store test class accuracy
test_no_obj_acc = load_data(LOG_PATH + f"test/no_object_accuracy/{curr_txt_file}.txt")    # store test no object accuracy
test_obj_acc = load_data(LOG_PATH + f"test/object_accuracy/{curr_txt_file}.txt")          # store test object accuracy


def my_plot(x, y, title, x_label, y_label, line_color, line_marker):
    plt.plot(x, y, color=line_color, marker=line_marker)
    
    # Initialize the store_path for all the figures, the folder_name should be the same as the log text file
    folder_name = curr_txt_file          # <class 'str'>
    store_path = FIG_PATH + folder_name  # 
    
    # If the folder doesn't exist, then we create that folder 
    if os.path.isdir(store_path) is False:
        print(f"creating {folder_name} folder to store the figures")
        os.makedirs(store_path)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # plt.show()

    # 
    fig_name = f"{folder_name}-{title}.png" 
    plt.savefig(store_path + "/" + fig_name, bbox_inches='tight') 
    plt.clf()             # clears the entire current figure 
    plt.close(plt.gcf())  # to avoid RuntimeWarning: More than 20 figures have been opened.


# it should follow the settings of the config.TEST_POINT, config.CHECK_TEST, or config.CHECK_MAP
test_point = 10  
def plot_mAP():
    # print(f"len(mAP): {len(mAP)}")
    # print(mAP)
    my_plot(
        x=[i*test_point for i in range(1, len(mAP) + 1)], 
        y=mAP, 
        title='mean-Average-Precision', 
        x_label='epochs', y_label='Area Under the Curve', 
        line_color='tab:red', line_marker='x',
    )


def plot_train_results():
    my_plot(
        x=[j for j in range(1, len(losses) + 1)], 
        y=losses, 
        title='training-loss-for-every-batch', 
        x_label='number of updates', y_label='loss value', 
        line_color='red', line_marker='',
    )
    my_plot(
        x=[j for j in range(1, len(mean_loss) + 1)], 
        y=mean_loss, 
        title='mean-training-loss-for-every-epoch',
        x_label='epochs', y_label='loss value', 
        line_color='red', line_marker='',
    )
    my_plot(
        x=[j for j in range(1, len(train_class_acc) + 1)], 
        y=train_class_acc, 
        title='train-class-accuracy',
        x_label='epochs', y_label='accuracy', 
        line_color='cornflowerblue', line_marker='',
    )
    my_plot(
        x=[j for j in range(1, len(train_no_obj_acc) + 1)], 
        y=train_no_obj_acc, 
        title='train-no-obj-accuracy',
        x_label='epochs', y_label='accuracy', 
        line_color='royalblue', line_marker='',
    )
    my_plot(
        x=[j for j in range(1, len(train_obj_acc) + 1)], 
        y=train_obj_acc, 
        title='train-obj-accuracy', 
        x_label='epochs', y_label='accuracy', 
        line_color='blue', line_marker='',
    )


def plot_test_results():
    my_plot(
        x=[j*test_point for j in range(1, len(test_class_acc) + 1)], 
        y=test_class_acc, 
        title='test-class-accuracy', 
        x_label='epochs', y_label='accuracy', 
        line_color='darkturquoise', line_marker='',
    )
    my_plot(
        x=[j*test_point for j in range(1, len(test_no_obj_acc) + 1)], 
        y=test_no_obj_acc, 
        title='test-no-obj-accuracy', 
        x_label='epochs', y_label='accuracy', 
        line_color='deepskyblue', line_marker='',
    )
    my_plot(
        x=[j*test_point for j in range(1, len(test_obj_acc) + 1)], 
        y=test_obj_acc, 
        title='test-obj-accuracy', 
        x_label='epochs', y_label='accuracy', 
        line_color='dodgerblue', line_marker='',
    )


def print_stats():
    print(f"-"*50)
    print(f"The stats of {curr_txt_file} training: ")
    print(f"-"*50)
    
    print(f"max mAP:  {max(mAP)}")
    print(f"mean mAP: {sum(mAP) / len(mAP)}\n")

    print(f"max training loss: {max(losses)}")
    print(f"min training loss: {min(losses)}\n")

    print(f"max training loss on average: {max(mean_loss)}")
    print(f"min training loss on average: {min(mean_loss)}\n")

    print(f"min training accuracy: {min(train_obj_acc)}")
    print(f"max training accuracy: {max(train_obj_acc)}\n")

    print(f"min testing accuracy: {min(test_obj_acc)}")
    print(f"max testing accuracy: {max(test_obj_acc)}")

    print(f"-"*50)


if __name__ == "__main__":

    print("plot training state!")

    # print_stats()
    # plot_mAP()
    # plot_train_results()
    # plot_test_results()


