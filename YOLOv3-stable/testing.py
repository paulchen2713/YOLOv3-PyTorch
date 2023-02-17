"""
@dependencies:
    env pt3.7
    python 3.7.13
    numpy==1.19.2
    pytorch==1.7.1
    torchaudio==0.7.2
    torchvision==0.8.2
"""

import torch
import numpy as np

num_anchors = 9
# S=[2, 4, 8] 
S=[13, 26, 52]
print("S[1:2] = ", S[1:2])
idx1, idx2 = 1, 1

targets = [torch.zeros((num_anchors // 3, s, s, 6)) for s in S]
# print(targets)

x, y = np.random.rand(2)
print(x, y)
i, j = int(S[idx1] * y), int(S[idx1] * x)
print(i, j)

anchor_taken = targets[idx1][idx2, i, j, 0] 
print(anchor_taken)

if not anchor_taken:
    print("anchor not taken")


x_cell, y_cell, width_cell, height_cell = np.random.rand(4)
print(type(x_cell), type(y_cell), type(width_cell), type(height_cell)) # <class 'numpy.float64'>
print(type([x_cell, y_cell, width_cell, height_cell])) # <class 'list'>
print(x_cell, y_cell, width_cell, height_cell)   # 0.38164115080294214 0.9454760442000583 0.6508773037358562 0.6399073486022664
print([x_cell, y_cell, width_cell, height_cell]) # [0.38164115080294214, 0.9454760442000583, 0.6508773037358562, 0.6399073486022664]

box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell]) 
print(type(box_coordinates)) # <class 'torch.Tensor'>
print(box_coordinates)       # tensor([0.3816, 0.9455, 0.6509, 0.6399], dtype=torch.float64)

