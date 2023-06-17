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
    '2023-04-27',   # 2023-04-27-1
    '2023-04-27-2',
    '2023-04-28',   # 2023-04-28-1
    '2023-04-28-2',
    '2023-04-28-3',
    '2023-04-29-1',
    '2023-04-29-2',
    '2023-04-30-1',
    '2023-04-30-2',
    '2023-05-01-1',
    '2023-05-01-2',
    '2023-05-01-3',
    '2023-05-02-1',
    '2023-05-02-2',
    '2023-05-02-3',
    '2023-05-02-4',
    '2023-05-03-1',
    '2023-05-03-2',
    '2023-05-04-1',
    '2023-05-04-2',
    '2023-05-05-1',
    '2023-05-06-1',
    '2023-05-07-1',
    '2023-05-08-1',
    '2023-05-09-1',
    '2023-05-10-1',
    '2023-05-11-1',
    '2023-05-12-1',
    '2023-05-13-1',
    '2023-05-14-1',
    '2023-05-15-1',
    '2023-05-16-1',
    '2023-05-16-2',
    '2023-05-16-3',
    '2023-05-16-4',
    '2023-05-17-1',
    '2023-05-17-2',
    '2023-05-18-2',
    '2023-05-18-3',
    '2023-05-19-1',
    '2023-05-19-2',
    '2023-05-20-1',
    '2023-05-20-2',
    '2023-05-21-1',
    '2023-05-22-1',
    '2023-05-22-2',
    '2023-06-05-1',
    '2023-06-07-1',  # NOTE the testpoint is 5
    '2023-06-08-1',
    '2023-06-08-3',
    '2023-06-09-1',
    '2023-06-10-1',
    '2023-06-13-2',
    '2023-06-14-1',
    '2023-06-15-1',
    '2023-06-15-2',
    '2023-06-16-1',
    '2023-06-16-2',
    '2023-06-17-1',
    '2023-06-17-2',
]

log_index = len(logs) - 1

weight_decay_indices = [7, 6, 5, 3]
learning_rate_01_04 = [9, 8, 3, 10]
learning_rate_05_08 = [11, 12, 13, 14]
learning_rate_09_12 = [15, 16, 17, 18]
learning_rate_13_16 = [19, 20, 21, 22]
learning_rate_17_20 = [23, 24, 25, 26]

# make sure we are using valid list subscripts
assert log_index < len(logs), f"len(logs) = {len(logs)}, the 'log_index' cannot exceed {len(logs) - 1}!"

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
├─2023-04-28-2
├─2023-04-28-3
├─2023-04-29-1
├─2023-04-29-2
├─2023-04-30-1
├─2023-04-30-2
├─2023-05-01-1
├─2023-05-01-2
├─2023-05-01-3
├─2023-05-02-1
├─2023-05-02-2
├─2023-05-02-3
├─2023-05-02-4
├─2023-05-03-1
├─2023-05-03-2
├─2023-05-04-1
├─different-learning-rate-results-1
├─different-learning-rate-results-2
├─different-learning-rate-results-3
├─different-learning-rate-results-4
├─different-weight-decay-results
└─training-time-comparison
"""


def get_file_content(file_name):
    with open(file_name) as f:
        return [line.strip() for line in f]


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as text_file:
        # print(f"current file: {file_path}")
        lines = text_file.readlines()
        data = []
        for line in lines:
            # adding a bunch of line.strip() to solve "ValueError: could not convert string to float"
            line = line.strip()
            line = line.strip("\n")
            line = line.strip("\t")
            line = line.strip("\"")
            data.append(float(line))
    return data


# empty containers
mAP_list = []
losses_list = []
train_acc_list = []
test_acc_list = []

# read the training and testing statistics results
curr_txt_file = logs[log_index]  # get the current file name

mAP = load_data(LOG_PATH + f"mAP/{curr_txt_file}.txt")                                   # store mean Average Precision

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


def my_plot(x: list, y: list, title, x_label, y_label, line_color, line_marker, overwrite=False):
    plt.plot(x, y, color=line_color, marker=line_marker)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Initialize the store_path for all the figures, the folder_name should be the same as the log text file
    folder_name = curr_txt_file          # <class 'str'>
    store_path = FIG_PATH + folder_name  # 
    
    # If the folder doesn't exist, then we create that folder 
    if os.path.isdir(store_path) is False:
        print(f"creating {folder_name} folder to store the figures")
        os.makedirs(store_path)
    
    fig_name = f"{folder_name}-{title}.png" 
    file_name = store_path + "/" + fig_name
    if os.path.isfile(f"{file_name}") is False:
        plt.savefig(file_name, bbox_inches='tight')
    elif overwrite is True:
        print(f"Note. overwriting '{fig_name}' ")
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()
    else:
        print(f"Note. '{fig_name}' already exists")
        plt.show()
    plt.clf()             # clears the entire current figure 
    plt.close(plt.gcf())  # to avoid RuntimeWarning: More than 20 figures have been opened.


# it should follow the settings of the config.TEST_POINT, config.CHECK_TEST, or config.CHECK_MAP
test_point = 5  # 5
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
    file_path = FIG_PATH + curr_txt_file + "/"
    file_name = file_path + f"stats-{curr_txt_file}.txt"
    if os.path.isfile(file_name) is True:
        print(f"Note. 'stats-{curr_txt_file}.txt' already exist")

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
    else:
        with open(file_name, "w") as txt_file:
            print(f"-"*50, file=txt_file)
            print(f"The stats of {curr_txt_file} training: ", file=txt_file)
            print(f"-"*50, file=txt_file)
            
            print(f"max mAP:  {max(mAP)}", file=txt_file)
            print(f"mean mAP: {sum(mAP) / len(mAP)}\n", file=txt_file)

            print(f"max training loss: {max(losses)}", file=txt_file)
            print(f"min training loss: {min(losses)}\n", file=txt_file)

            print(f"max training loss on average: {max(mean_loss)}", file=txt_file)
            print(f"min training loss on average: {min(mean_loss)}\n", file=txt_file)

            print(f"min training accuracy: {min(train_obj_acc)}", file=txt_file)
            print(f"max training accuracy: {max(train_obj_acc)}\n", file=txt_file)

            print(f"min testing accuracy: {min(test_obj_acc)}", file=txt_file)
            print(f"max testing accuracy: {max(test_obj_acc)}", file=txt_file)

            print(f"-"*50, file=txt_file)
        
        print(f"saving stats to {file_path} as stats-{curr_txt_file}.txt")


def load_multiple_train_results(indices: list):
    mAP_list = []
    losses_list = []
    train_acc_list = []
    test_acc_list = []

    for index in indices:
        curr_mAP = load_data(LOG_PATH + f"mAP/{logs[index]}.txt")
        mAP_list.append(curr_mAP)

        curr_losses = load_data(LOG_PATH + f"train/losses/{logs[index]}.txt")
        losses_list.append(curr_losses)

        curr_train_acc = load_data(LOG_PATH + f"train/object_accuracy/{logs[index]}.txt")
        train_acc_list.append(curr_train_acc)

        curr_test_acc = load_data(LOG_PATH + f"test/object_accuracy/{logs[index]}.txt")
        test_acc_list.append(curr_test_acc)
    
    return mAP_list, losses_list, train_acc_list, test_acc_list


folder_index = 6
def plot_diff_setting(x: list, data: list, title, x_label, y_label, folder_name, mode, overwrite=False):
    fig_name = f"{title}-with-different-{mode}"

    if mode == 'weight-decay':
        for i, curr in enumerate(data):
            # print(len(x), len(curr))
            assert len(x) == len(curr), f"data length mismatch: {len(x)}, {len(curr)}"
            plt.plot(x, curr, label=f"WEIGHT_DECAY = 1e-{i+1}")
    elif mode == 'learning-rate':
        for j, curr in enumerate(data):
            # print(len(x), len(curr))
            assert len(x) == len(curr), f"data length mismatch: {len(x)}, {len(curr)}"
            plt.plot(x, curr, label=f"LEARNING_RATE = {j+1+4*(folder_index - 1)}e-5")
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(fig_name)
    plt.legend()
    
    store_path = FIG_PATH + folder_name
    # If the folder doesn't exist, then we create that folder 
    if os.path.isdir(store_path) is False:
        print(f"creating {folder_name} folder to store the figures")
        os.makedirs(store_path)

    file_name = store_path + "/" + fig_name
    if os.path.isfile(f"{file_name}.png") is False:
        plt.savefig(f"{file_name}.png", bbox_inches='tight') 
    elif overwrite is True:
        print(f"Note. overwriting '{fig_name}.png' ")
        plt.savefig(f"{file_name}.png", bbox_inches='tight') 
        plt.show()
    else:
        print(f"Note. '{fig_name}.png' already exists")
        plt.show()
    plt.clf()             # clears the entire current figure 
    plt.close(plt.gcf())  # to avoid RuntimeWarning: More than 20 figures have been opened.


def plot_multi_results(mode: str, overwrite=False):
    folder_name = f""
    if mode == 'learning-rate':
        folder_name = f'different-{mode}-results-{folder_index}'
    else:  # mode == 'weight-decay'
        folder_name = f'different-{mode}-results'
    
    plot_diff_setting(
        x=[j*test_point for j in range(1, len(mAP_list[0]) + 1)],
        data=mAP_list,
        title='mAP',
        x_label='epochs',
        y_label='Area Under the Curve',
        folder_name=folder_name,
        mode=mode,
        overwrite=overwrite,
    )
    plot_diff_setting(
        x=[j for j in range(1, len(losses_list[0]) + 1)],
        data=losses_list,
        title='losses',
        x_label='number of updates',
        y_label='loss value',
        folder_name=folder_name,
        mode=mode,
        overwrite=overwrite,
    )
    plot_diff_setting(
        x=[j for j in range(1, len(train_acc_list[0]) + 1)], 
        data=train_acc_list, 
        title='train-accuracy', 
        x_label='epochs', 
        y_label='accuracy', 
        folder_name=folder_name,
        mode=mode,
        overwrite=overwrite,
    )
    plot_diff_setting(
        x=[j*test_point for j in range(1, len(test_acc_list[0]) + 1)], 
        data=test_acc_list, 
        title='test-accuracy', 
        x_label='epochs', 
        y_label='accuracy', 
        folder_name=folder_name,
        mode=mode,
        overwrite=overwrite,
    )


def plot_training_duration(overwrite=False):
    time_with_diff_lr = [7.5511, 7.2838, 7.2117, 7.1383, 7.1785, 
                         7.0542, 7.1015, 6.7780, 5.5800, 5.7350, 
                         7.0366, 7.1689, 6.8200, 5.8219, 5.5819,
                         8.4758, 7.1341, 7.1676, 7.4228, 8.4733,]
    time_with_diff_wd = [7.1676, 7.7900, 6.2753, 7.2117, ]

    lr = [f"{i}" for i in range(1, len(time_with_diff_lr) + 1)]
    wd = [f'1e-{i+1}' for i in range(len(time_with_diff_wd))]

    store_path = FIG_PATH + f'training-time-comparison/'

    # plot different learning rate vs training duration
    plt.plot(lr, time_with_diff_lr)
    # plt.xscale('log')
    plt.xlabel('learning rate (e-5)')
    plt.ylabel('training duration (hour)')
    title1 = f'learning-rate-vs-training-duration'
    plt.title(title1)
    
    file_name = store_path + f"{title1}.png"
    if os.path.isfile(f"{file_name}") is False:
        plt.savefig(file_name, bbox_inches='tight')
    elif overwrite is True:
        print(f"Note. overwriting '{title1}.png' ")
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()
    else:
        print(f"Note. '{title1}.png' already exists")
        plt.show()
    plt.clf()             # clears the entire current figure 
    plt.close(plt.gcf())

    # plot different weight decay vs training duration
    plt.plot(wd, time_with_diff_wd, color='b')
    plt.xlabel('weight decay')
    plt.ylabel('training duration (hour)')
    title2 = f'weight-decay-vs-training-duration'
    plt.title(title2)

    file_name = store_path + f"{title2}.png"
    if os.path.isfile(f"{file_name}") is False:
        plt.savefig(file_name, bbox_inches='tight')
    elif overwrite is True:
        print(f"Note. overwriting '{title2}.png' ")
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()
    else:
        print(f"Note. '{title2}.png' already exists")
        plt.show()
    plt.clf()             # clears the entire current figure 
    plt.close(plt.gcf())



if __name__ == "__main__":

    print("plot training state!")

    plot_mAP()
    plot_train_results()
    plot_test_results()
    print_stats()

    # mAP_list, losses_list, train_acc_list, test_acc_list = load_multiple_train_results(indices=weight_decay_indices)
    # plot_multi_results(mode='weight-decay', overwrite=False)
    
    # mAP_list, losses_list, train_acc_list, test_acc_list = load_multiple_train_results(indices=learning_rate_17_20)
    # plot_multi_results(mode='learning-rate', overwrite=False)
    
    # plot_training_duration(overwrite=False)
    
