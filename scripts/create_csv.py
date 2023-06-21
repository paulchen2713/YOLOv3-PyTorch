# -*- coding: utf-8 -*-
"""
Created on Thu Mar 2 16:23:09 2023

@patch:
    2023.02.26
@author: Paul
@file: create_csv.py
@dependencies:
    env pt3.7 (my PC)
    python 3.7.13
    pytorch==1.7.1     py3.7_cuda110_cudnn8_0 pytorch
    torchaudio==0.7.2  py37 pytorch
    torchvision==0.8.2 py37_cu110 pytorch

Generate .csv or .txt files for training and testing use 
"""

import os
import csv
import pandas as pd
import random
import time
from datetime import date

DATASET = f"D:/Datasets/RADA/RD_JPG/"


def create_csv(num_train, total):
    train_file_name = DATASET + "csv_files/original_csv/" + f"train.csv"
    if os.path.isfile(f"{train_file_name}") is False:
        print(f"Creating '{train_file_name}' with {num_train} samples")
        with open(train_file_name, "w") as train_file:
            for i in range(1, num_train + 1):
                print(f"{i}.jpg,{i}.txt", file=train_file)

    test_file_name = DATASET + "csv_files/original_csv/" + f"test.csv"
    if os.path.isfile(f"{test_file_name}") is False:
        print(f"Creating '{test_file_name}' with {total - num_train} samples")
        with open(test_file_name, "w") as test_file:
            for i in range(num_train + 1, total + 1):
                print(f"{i}.jpg,{i}.txt", file=test_file)

    if os.path.isfile(f"{train_file_name}") is True and os.path.isfile(f"{test_file_name}") is True:
        print(f"Both '{train_file_name}' and '{test_file_name}' are already exist!\n")


indices = [i for i in range(1, 7193 + 1)]
def random_csv(num_train, num_test):
    # print(len(indices)) # 7193
    random.shuffle(indices)

    rand_train_file_name = DATASET + "csv_files/rand_csv/" + f"train.csv"
    rand_test_file_name  = DATASET + "csv_files/rand_csv/" + f"test.csv"

    with open(rand_train_file_name, "w") as train_file:
        for i in indices[0:num_train]:
            print(f"{i}.jpg,{i}.txt", file=train_file)
    
    with open(rand_test_file_name, "w") as test_file:
        for i in indices[-num_test::1]:
            print(f"{i}.jpg,{i}.txt", file=test_file)


def equal_splits_csv(split):
    print(f"taking the {split} split as the test samples")

    TOTAL_SPLIT = 6 # has to be 6
    # split = 0       # 0, 1, 3, 4, 5

    train_file_name = DATASET + f"csv_files/equal_split_csv/{split}/" + f"train.csv"
    test_file_name  = DATASET + f"csv_files/equal_split_csv/{split}/" + f"test.csv"
    
    if os.path.isfile(f"{test_file_name}") is True:
        print(f"the '{test_file_name}' is already exits!")
        return

    for i in range(1, 7193 + 1):
        print(i)
        if i % TOTAL_SPLIT == split:
            with open(test_file_name, "a") as test_file:
                print(f"{i}.jpg,{i}.txt", file=test_file)
        else:
            with open(train_file_name, "a") as train_file:
                print(f"{i}.jpg,{i}.txt", file=train_file)


skip_list1 = [
    [i for i in range(273, 288 + 1)], 
    [i for i in range(857, 867 + 1)], 
    [i for i in range(1405, 1416 + 1)], 
    [i for i in range(1628, 1632 + 1)], 
    [i for i in range(1654, 1678 + 1)], 
    [i for i in range(1702, 1729 + 1)], 
    [i for i in range(1758, 1782 + 1)], 
    [i for i in range(1803, 1826 + 1)], 
    [i for i in range(1862, 1904 + 1)], 
    [i for i in range(1955, 1991 + 1)], 
    [i for i in range(2305, 2311 + 1)], 
    [i for i in range(4812, 4867 + 1)], 
    [i for i in range(4985, 5044 + 1)], 
    [i for i in range(5147, 5232 + 1)], 
    [i for i in range(5338, 5434 + 1)], 
    [i for i in range(5506, 5563 + 1)], 
    [i for i in range(5778, 5842 + 1)], 
    [i for i in range(6319, 6407 + 1)], 
    [i for i in range(6599, 6634 + 1)], 
    [i for i in range(7071, 7114 + 1)], 
]
skip_list2 = [
    273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1628, 1629, 1630, 1631, 1632, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 4812, 4813, 4814, 4815, 4816, 4817, 4818, 4819, 4820, 4821, 4822, 4823, 4824, 4825, 4826, 4827, 4828, 4829, 4830, 4831, 4832, 4833, 4834, 4835, 4836, 4837, 4838, 4839, 4840, 4841, 4842, 4843, 4844, 4845, 4846, 4847, 4848, 4849, 4850, 4851, 4852, 4853, 4854, 4855, 4856, 4857, 4858, 4859, 4860, 4861, 4862, 4863, 4864, 4865, 4866, 4867, 4985, 4986, 4987, 4988, 4989, 4990, 4991, 4992, 4993, 4994, 4995, 4996, 4997, 4998, 4999, 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010, 5011, 5012, 5013, 5014, 5015, 5016, 5017, 5018, 5019, 5020, 5021, 5022, 5023, 5024, 5025, 5026, 5027, 5028, 5029, 5030, 5031, 5032, 5033, 5034, 5035, 5036, 5037, 5038, 5039, 5040, 5041, 5042, 5043, 5044, 5147, 5148, 5149, 5150, 5151, 5152, 5153, 5154, 5155, 5156, 5157, 5158, 5159, 5160, 5161, 5162, 5163, 5164, 5165, 5166, 5167, 5168, 5169, 5170, 5171, 5172, 5173, 5174, 5175, 5176, 5177, 5178, 5179, 5180, 5181, 5182, 5183, 5184, 5185, 5186, 5187, 5188, 5189, 5190, 5191, 5192, 5193, 5194, 5195, 5196, 5197, 5198, 5199, 5200, 5201, 5202, 5203, 5204, 5205, 5206, 5207, 5208, 5209, 5210, 5211, 5212, 5213, 5214, 5215, 5216, 5217, 5218, 5219, 5220, 5221, 5222, 5223, 5224, 5225, 5226, 5227, 5228, 5229, 5230, 5231, 5232, 5338, 5339, 5340, 5341, 5342, 5343, 5344, 5345, 5346, 5347, 5348, 5349, 5350, 5351, 5352, 5353, 5354, 5355, 5356, 5357, 5358, 5359, 5360, 5361, 5362, 5363, 5364, 5365, 5366, 5367, 5368, 5369, 5370, 5371, 5372, 5373, 5374, 5375, 5376, 5377, 5378, 5379, 5380, 5381, 5382, 5383, 5384, 5385, 5386, 5387, 5388, 5389, 5390, 5391, 5392, 5393, 5394, 5395, 5396, 5397, 5398, 5399, 5400, 5401, 5402, 5403, 5404, 5405, 5406, 5407, 5408, 5409, 5410, 5411, 5412, 5413, 5414, 5415, 5416, 5417, 5418, 5419, 5420, 5421, 5422, 5423, 5424, 5425, 5426, 5427, 5428, 5429, 5430, 5431, 5432, 5433, 5434, 5506, 5507, 5508, 5509, 5510, 5511, 5512, 5513, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525, 5526, 5527, 5528, 5529, 5530, 5531, 5532, 5533, 5534, 5535, 5536, 5537, 5538, 5539, 5540, 5541, 5542, 5543, 5544, 5545, 5546, 5547, 5548, 5549, 5550, 5551, 5552, 5553, 5554, 5555, 5556, 5557, 5558, 5559, 5560, 5561, 5562, 5563, 5778, 5779, 5780, 5781, 5782, 5783, 5784, 5785, 5786, 5787, 5788, 5789, 5790, 5791, 5792, 5793, 5794, 5795, 5796, 5797, 5798, 5799, 5800, 5801, 5802, 5803, 5804, 5805, 5806, 5807, 5808, 5809, 5810, 5811, 5812, 5813, 5814, 5815, 5816, 5817, 5818, 5819, 5820, 5821, 5822, 5823, 5824, 5825, 5826, 5827, 5828, 5829, 5830, 5831, 5832, 5833, 5834, 5835, 5836, 5837, 5838, 5839, 5840, 5841, 5842, 6319, 6320, 6321, 6322, 6323, 6324, 6325, 6326, 6327, 6328, 6329, 6330, 6331, 6332, 6333, 6334, 6335, 6336, 6337, 6338, 6339, 6340, 6341, 6342, 6343, 6344, 6345, 6346, 6347, 6348, 6349, 6350, 6351, 6352, 6353, 6354, 6355, 6356, 6357, 6358, 6359, 6360, 6361, 6362, 6363, 6364, 6365, 6366, 6367, 6368, 6369, 6370, 6371, 6372, 6373, 6374, 6375, 6376, 6377, 6378, 6379, 6380, 6381, 6382, 6383, 6384, 6385, 6386, 6387, 6388, 6389, 6390, 6391, 6392, 6393, 6394, 6395, 6396, 6397, 6398, 6399, 6400, 6401, 6402, 6403, 6404, 6405, 6406, 6407, 6599, 6600, 6601, 6602, 6603, 6604, 6605, 6606, 6607, 6608, 6609, 6610, 6611, 6612, 6613, 6614, 6615, 6616, 6617, 6618, 6619, 6620, 6621, 6622, 6623, 6624, 6625, 6626, 6627, 6628, 6629, 6630, 6631, 6632, 6633, 6634, 7071, 7072, 7073, 7074, 7075, 7076, 7077, 7078, 7079, 7080, 7081, 7082, 7083, 7084, 7085, 7086, 7087, 7088, 7089, 7090, 7091, 7092, 7093, 7094, 7095, 7096, 7097, 7098, 7099, 7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109, 7110, 7111, 7112, 7113, 7114,
]


def get_dirty_and_clean_list():
    dirty_list = []
    size = 0
    for lst in skip_list1:
        # print(f"{len(lst)}: ") 
        size += len(lst)
        for idx in lst:
            dirty_list.append(idx)
            # print(f"{idx}", end=" ") 
    # print(f"total skip: {size}")  # total skip: 824 

    # for idx in total:
    #     print(idx, end=", ")
    # print(f"")
    # print(len(total))  # 824
    # print(total)

    clean_list = []
    for idx in indices:
        if idx in skip_list2: continue
        # print(idx, end=" ")
        clean_list.append(idx)
    # print(len(clean_list))  # 6369
    return dirty_list, clean_list


TOTAL_SPLIT = 8  # NOTE 
def clean_csv(input_list, split):
    # print(f"Creating csv files without dirty data, taking the {split} split as the test samples")

    store_path = DATASET + f"csv_files/clean_csv/{TOTAL_SPLIT}-fold/{split}/"

    if os.path.isdir(store_path) is False:
        print(f"creating /{TOTAL_SPLIT}-fold/{split} folder to store the csv files")
        os.makedirs(store_path)

    test_file_name  = store_path + f"test.csv"
    train_file_name = store_path + f"train.csv"
    
    if os.path.isfile(f"{test_file_name}") is True:
        print(f"the '{test_file_name}' is already exits!")

        temp_test  = pd.read_csv(test_file_name)
        temp_train = pd.read_csv(train_file_name)

        print(f"the number of test and train samples for {split} split: {len(temp_test) + 1}, {len(temp_train) + 1}")
        return len(temp_test) + 1, len(temp_train) + 1

    # print(f"total number of data: {len(input_list)}")  # total number of data: 6369
    num_of_test, num_of_train = 0, 0
    for i in input_list:
        if i % TOTAL_SPLIT == split:
            num_of_test += 1
            with open(test_file_name, "a") as test_file:
                print(f"{i}.jpg,{i}.txt", file=test_file)
        else:
            num_of_train += 1
            with open(train_file_name, "a") as train_file:
                print(f"{i}.jpg,{i}.txt", file=train_file)
    
    print(f"the number of test and train samples for {split} split: {num_of_test}, {num_of_train} \n")
    return num_of_test, num_of_train


def dirty_and_clean_csv(dirty_indices, clean_indices):
    root_path = DATASET + f"csv_files/" 
    
    if os.path.isfile(root_path + f"clean.csv") is False:
        with open(root_path + f"clean.csv", "a") as csv_file:
            for i in clean_indices:
                print(f"{i}.jpg,{i}.txt", file=csv_file)
    else:
        print(f"the '{root_path}clean.csv' is already exists!")
    
    if os.path.isfile(root_path + f"dirty.csv") is False:
        with open(root_path + f"dirty.csv", "a") as csv_file:
            for i in dirty_indices:
                print(f"{i}.jpg,{i}.txt", file=csv_file)
    else:
        print(f"the '{root_path}dirty.csv' is already exists!")


def make_folders(clean_indices):
    for index in range(0, TOTAL_SPLIT):
        # equal_splits_csv(split=index)
        num_of_test, num_of_train = clean_csv(input_list=clean_indices, split=index)
        
        store_path = DATASET + f"csv_files/clean_csv/{TOTAL_SPLIT}-fold/{index}/"
        if os.path.isdir(store_path + "tested/") is False:
            os.makedirs(store_path + "tested/")
        if os.path.isfile(store_path + f"README.txt") is False:
            with open(store_path + f"README.txt", "w") as txt_file:
                print(f"##split {index}", file=txt_file)
                print(f"number of test samples:  {num_of_test}", file=txt_file)
                print(f"number of train samples: {num_of_train}", file=txt_file)
        else: 
            continue
    print(f"the '{TOTAL_SPLIT}-fold' is already exists!")


if __name__ == "__main__":
    tic = time.perf_counter()

    num_train, total = 6000, 7193
    # create_csv(num_train=num_train, total=total)
    # random_csv(num_train=num_train, num_test=(total-num_train))

    dirty_indices, clean_indices = get_dirty_and_clean_list()
    # print(dirty_indices == skip_list2)  # True
    # print(len(clean_indices))           # 6369

    make_folders(clean_indices=clean_indices)

    dirty_and_clean_csv(dirty_indices=dirty_indices, clean_indices=clean_indices)

    
    

    toc = time.perf_counter()
    duration = toc - tic
    print(f"duration: {duration:0.4f} seconds")

