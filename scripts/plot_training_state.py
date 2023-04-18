# 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import date as date_function

# print(f"{date_function.today()}")

# the path where we placed the log files
PATH = 'D:/Datasets/RADA/RD_JPG/training_logs/'

folders = ['mAP', 'train', 'test']
data_folders = ['losses', 'mean_loss', 'class_accuracy', 'no_object_accuracy', 'object_accuracy']

logs = [
    '2023-03-23.txt',
    '2023-04-07.txt',
    '2023-04-15.txt',
    '2023-04-16.txt',
]
log_index = 1

# the file tree structure:
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


# training statistics
actual_loss = []  # the actual loss for every batch, the total number would be 'epoch x split'
mean_loss = []    # store the average loss for every epoch 
train_class_acc = []   # store train class accuracy
train_no_obj_acc = []  # store train object accuracy
train_obj_acc = []     # store train object accuracy


# testing statistics
test_class_acc = []    # store test class accuracy
test_no_obj_acc = []   # store test object accuracy
test_obj_acc = []      # store test object accuracy
mAP = []

# 
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as text_file:
        # print(f"current file: {file_path}")
        data = []
        for line in text_file:
            data.append(float(line))
    return data


def load_losses(file_path):
    with open(file_path, "r", encoding="utf-8") as text_file:
        # print(f"current file: {file_path}")
        data = []
        for line in text_file:
            data.append(line)
    return data


def my_plot(x, y, title, y_label, line_color, line_marker):
    plt.plot(x, y, color=line_color, marker=line_marker)
    
    # for xy in zip(x, y):
    #     plt.annotate('(%d, %.4f)' % xy, xy=xy)
    
    plt.xlabel('epochs')
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()






# read and plot the training statistics results
mAP = load_data(PATH + f"mAP/{logs[log_index]}")
losses = load_data(PATH + f"train/losses/{logs[log_index]}")
mean_loss = load_data(PATH + f"train/mean_loss/{logs[log_index]}")
train_class_acc = load_data(PATH + f"train/class_accuracy/{logs[log_index]}")
train_no_obj_acc = load_data(PATH + f"train/no_object_accuracy/{logs[log_index]}")
train_obj_acc = load_data(PATH + f"train/object_accuracy/{logs[log_index]}")


# read and plot mAP 
def plot_mAP():
    # print(f"len(mAP): {len(mAP)}")
    # print(mAP)
    my_plot([i*100 for i in range(1, len(mAP) + 1)], mAP, 'mean Average Precision', 'Area Under the Curve', 'tab:red', 'x')


def plot_train_results():
    # my_plot([j for j in range(1, len(losses) + 1)], losses, 'training loss for every batch', 'loss value', 'red', '')
    my_plot([j for j in range(1, len(mean_loss) + 1)], mean_loss, 'mean training loss', 'loss value', 'red', '')
    my_plot([j for j in range(1, len(train_class_acc) + 1)], train_class_acc, 'train class accuracy', 'accuracy', 'cornflowerblue', '')
    my_plot([j for j in range(1, len(train_no_obj_acc) + 1)], train_no_obj_acc, 'train no obj accuracy', 'accuracy', 'royalblue', '')
    my_plot([j for j in range(1, len(train_obj_acc) + 1)], train_obj_acc, 'train obj accuracy', 'accuracy', 'blue', '')


# read and plot the testing statistics results
test_class_acc = load_data(PATH + f"test/class_accuracy/{logs[log_index]}")
test_no_obj_acc = load_data(PATH + f"test/no_object_accuracy/{logs[log_index]}")
test_obj_acc = load_data(PATH + f"test/object_accuracy/{logs[log_index]}")

test_point = 10
def plot_test_results():
    my_plot([j*test_point for j in range(1, len(test_class_acc) + 1)], test_class_acc, 'test class accuracy', 'accuracy', 'darkturquoise', '')
    my_plot([j*test_point for j in range(1, len(test_no_obj_acc) + 1)], test_no_obj_acc, 'test no obj accuracy', 'accuracy', 'deepskyblue', '')
    my_plot([j*test_point for j in range(1, len(test_obj_acc) + 1)], test_obj_acc, 'test obj accuracy', 'accuracy', 'dodgerblue', '')


print(f"max mAP: {max(mAP)}")
print(f"mean mAP: {sum(mAP) / len(mAP)}")

print(f"max training loss: {max(losses)}")
print(f"min training loss: {min(losses)}")

print(f"max training loss on average: {max(mean_loss)}")
print(f"min training loss on average: {min(mean_loss)}")

print(f"min training accuracy: {min(train_obj_acc)}")
print(f"max training accuracy: {max(train_obj_acc)}")

print(f"min testing accuracy: {min(test_obj_acc)}")
print(f"max testing accuracy: {max(test_obj_acc)}")


my_plot([j for j in range(1, len(losses) + 1)], losses, 'training loss for every batch', 'loss value', 'red', '')

# plot_mAP()
# plot_train_results()
# plot_test_results()



