
import numpy as np
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt

# the path where we placed the log files
PATH = 'D:/Datasets/RADA/RD_JPG/training_logs/'

# log file names
logs = [
    'log_0316.txt',
    'log_0317.txt',
    'log_0318.txt',
]

# set the file path for the training logs
file_index = 0
file_path = PATH + f"{logs[file_index]}"
data = []  # the buffer for storing all data in the log files

loss = []       # store training loss
train_acc = []  # store train object accuracy
valid_acc = []  # store test object accuracy
train_class_acc = []  # store train class accuracy
valid_class_acc = []  # store test class accuracy


def load_data():
    with open(file_path, "r", encoding="utf-8") as text_file:
        # print(f"current file: {file_path}")
        for line in text_file:
            data.append(line)
            # print(line)
    print(f"all the data from {file_path} has been loaded in 'data' list")


# print(len(data[1])) # 179
# print(len(data)) # 768
def read_data():
    train_flag, valid_flag = False, False
    train_class_flag, valid_class_flag = False, False
    for i in range(len(data)):
        # Training Loss
        if data[i][167:171] == "loss":
            loss.append(float(data[i][172:176]))
        if data[i][168:172] == "loss":
            loss.append(float(data[i][173:176]))
        if data[i][170:174] == "loss":  # line 312
            loss.append(float(data[i][175]))
        
        # Train Obj Accuracy and Class Accuracy
        if data[i][3:8] == "Train":
            train_flag, train_class_flag = True, True
        if train_flag == True and data[i][0:3] == "Obj":
            train_acc.append(float(data[i][17:22]))
            train_flag = False
        if train_class_flag == True and data[i][0:5] == "Class":
            train_class_acc.append(float(data[i][19:24]))
            train_class_flag = False

        # Test Obj Accuracy and Class Accuracy
        if data[i][3:7] == "Test":
            valid_flag, valid_class_flag = True, True
        if valid_flag == True and data[i][0:3] == "Obj":
            valid_acc.append(float(data[i][17:22]))
            valid_flag = False
        if valid_class_flag == True and data[i][0:5] == "Class":
            valid_class_acc.append(float(data[i][19:24]))
            valid_class_flag = False


def print_vals():
    # print(logs[file_index][4:8])
    with open(PATH + f"{logs[file_index][4:8]}/{logs[file_index][4:8]}.txt", "w") as txt_file:
        print(f"training loss = {loss}", file=txt_file)
        print(f"num of loss samples: {len(loss)}", file=txt_file)        # 100

        print(f"train class accuracy = {train_class_acc}", file=txt_file)
        print(f"train obj accuracy = {train_acc}", file=txt_file)
        print(f"num of train samples: {len(train_acc)}", file=txt_file)  # 100

        print(f"test class accuracy = {valid_class_acc}", file=txt_file)
        print(f"test obj accuracy = {valid_acc}", file=txt_file)
        print(f"num of test samples: {len(valid_acc)}", file=txt_file)   # 11


def my_plot(x1, y1, title, line_color, line_marker):
    plt.plot(x1, y1, color=line_color, marker=line_marker)
    plt.xlabel('epochs')

    if title == "test accuracy":
        plt.plot([i for i in range(1, len(valid_class_acc) + 1)], valid_class_acc, color='green', marker='+')
        plt.ylabel('object accuracy')
        plt.title(title)
        plt.show()

    elif title == "train accuracy":
        plt.plot([i for i in range(1, len(train_class_acc) + 1)], train_class_acc, color='green', marker='+')
        plt.ylabel('object accuracy')
        plt.title(title)
        plt.show()

    elif title == "training loss":
        plt.ylabel('loss')
        plt.title(title)
        plt.show()


def plot_results():
    my_plot([i for i in range(1, len(loss) + 1)], loss, 'training loss', 'red', 'o')
    my_plot([i for i in range(1, len(train_acc) + 1)], train_acc, 'train accuracy', 'blue', 'o')
    my_plot([i for i in range(1, len(valid_acc) + 1)], valid_acc, 'test accuracy', 'blue', 'o')


if __name__ == '__main__':
    print(f"reading log files")

    load_data()
    read_data()
    print_vals()
    plot_results()




