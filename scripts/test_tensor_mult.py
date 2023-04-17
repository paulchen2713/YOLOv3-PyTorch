import numpy as np

def init_tensors(size=9, m=5, n=5):
    # sub_arr = np.zeros((m, n)).tolist()
    sub_arr = np.random.randint(1, 10, [m, n])
    sub_arr = np.array(sub_arr)
    # print(sub_arr)
    # print(sub_arr.shape)  # (5, 5)

    arr = [sub_arr] * size
    arr = np.array(arr)
    # print(arr[0].shape)   # (5, 5)
    # print(arr.shape)      # (9000, 5, 5)
    return arr


A = init_tensors(10, 3, 2)
B = init_tensors(10, 2, 3)

print(f"A.shape: {A.shape}")
print(f"B.shape: {B.shape}")

# print(f"A: ")
# print(f"{A}")
# print(f"B: ")
# print(f"{B}")

result = A @ B
print(f"result.shape: {result.shape}")

print(f"\nA[0, :, :]: ")
print(f"{A[0, :, :]}")
print(f"A[0].shape: {A[0].shape}")

print(f"\nB[0, :, :]: ")
print(f"{B[0, :, :]}")
print(f"B[0].shape: {B[0].shape}")

print(f"\nA[0] @ B[0]: ")
temp = A[0, :, :] @ B[0, :, :]
print(f"{temp}")
# print(type(temp))         # <class 'numpy.ndarray'>


print(f"\nresult: ")
print(f"{result[0, :, :]}") 
# print(type(result[0]))    # <class 'numpy.ndarray'>


