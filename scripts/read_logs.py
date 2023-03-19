import numpy as np
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt

def my_plot(x1, y1, title, line_color, line_marker):
    plt.plot(x1, y1, color=line_color, marker=line_marker)
    plt.xlabel('epochs')

    if title == "test accuracy":
        plt.ylabel('object accuracy')
        plt.title(title)
        plt.show()

    elif title == "train accuracy":
        plt.ylabel('object accuracy')
        plt.title(title)
        plt.show()

    elif title == "training loss":
        plt.ylabel('loss')
        plt.title(title)
        plt.show()


logs = [
    'log_0316.txt',
    'log_0317.txt',
    'log_0318.txt',
]

# set the file path for the training logs
file_index = 0
file_path = f"D:/Datasets/RADA/RD_JPG/training_logs/{logs[file_index]}"
data = [] 
with open(file_path, "r", encoding="utf-8") as text_file:
    # print(f"current file: {file_path}")
    for line in text_file:
        data.append(line)
        # print(line)

loss = []
train_acc = []
valid_acc = []

# print(len(data[1])) # 179
# print(len(data)) # 768
train_flag, valid_flag = False, False
for i in range(len(data)):
    # Training Loss
    if data[i][167:171] == "loss":
        loss.append(float(data[i][172:176]))
    if data[i][168:172] == "loss":
        loss.append(float(data[i][173:176]))
    if data[i][170:174] == "loss":  # line 312
        loss.append(float(data[i][175]))
    # Training Obj Accuracy
    if data[i][3:8] == "Train":
        train_flag = True
    if train_flag == True and data[i][0:3] == "Obj":
        train_acc.append(float(data[i][17:22]))
        train_flag = False
    # Testing Obj Accuracy
    if data[i][3:7] == "Test":
        valid_flag = True
    if valid_flag == True and data[i][0:3] == "Obj":
        valid_acc.append(float(data[i][17:22]))
        valid_flag = False

def print_vals():
    print(f"training loss = {loss}")
    print(f"num of loss samples: {len(loss)}")        # 100

    print(f"train obj accuracy = {train_acc}")
    print(f"num of train samples: {len(train_acc)}")  # 100

    print(f"test obj accuracy = {valid_acc}")
    print(f"num of test samples: {len(valid_acc)}")   # 11

def plot_results():
    my_plot([i for i in range(1, len(loss) + 1)], loss, 'training loss', 'red', 'o')
    my_plot([i for i in range(1, len(train_acc) + 1)], train_acc, 'train accuracy', 'blue', 'o')
    my_plot([i for i in range(1, len(valid_acc) + 1)], valid_acc, 'test accuracy', 'green', 'o')


# print_vals()
# plot_results()




