import numpy as np
from functools import cmp_to_key

def init_tensors(n=5, size=9000):
    sub_arr = np.zeros((n, n)).tolist()
    sub_arr = np.array(sub_arr)
    # print(sub_arr)
    print(sub_arr.shape)  # (5, 5)

    arr = [sub_arr] * size
    arr = np.array(arr)
    print(arr[0].shape)   # (5, 5)
    print(arr.shape)      # (9000, 5, 5)
    return arr


# data = [
#     (0.25288722826086957, 0.05166227921195651), (0.038317101353581864, 0.015974486216573144), (0.5012400793650792, 0.0916625330687831), 
#     (0.1640947386519945, 0.030481645460797804), (0.10124916443850268, 0.024327895220588237), (0.35412644787644776, 0.09159326737451738), 
#     (0.21300154320987646, 0.097752700617284), (0.051093155893536135, 0.07215423003802279), (0.14553571428571427, 0.08112723214285716),
# ]

data = [
    (0.424, 0.095), (0.040, 0.048), (0.121, 0.025), 
    (0.219, 0.041), (0.016, 0.016), (0.219, 0.097), 
    (0.039, 0.009), (0.125, 0.073), (0.058, 0.019),
]

data = np.array(data) # convert list of tuples into np.array of tuples
print(data.shape)     # (9, 2)
data = data.tolist()  # convert it back to list of tuples
print(len(data))      # 9
print(len(data[0]))   # 2
print(type(data))     # <class 'list'>
print(type(data[0]))  # <class 'list'>

print(f"before sort: ")
for d in data:
    w, h = d
    # Print out the width and height (w, h) of an anchor along with its cross-sectional area multiplied by 10000. 
    # This will help us distinguish between high and low anchor values easily.
    print(f"{d}   {w*h*10000}") 

def cmp_by_area(a, b):
    area_a = a[0] * a[1]
    area_b = b[0] * b[1]
    if area_a == area_b:
        return 0
    elif area_a < area_b:
        return -1 
    else: 
        return 1

cmp_key = cmp_to_key(cmp_by_area)
data.sort(key=cmp_key)

print(f"after sort: ")
for d in data:
    w, h = d
    # Print out the width and height (w, h) of an anchor along with its cross-sectional area multiplied by 10000. 
    # This will help us distinguish between high and low anchor values easily.
    print(f"{d}   {w*h*10000}")

for d in data:
    w, h = d
    print(f"({w:0.3f}, {h:0.3f})", end=', ')
    