import numpy as np
from matplotlib import pyplot as plt

def f(x):
   return (2 * x) / (1 + x)

def plot_iou_curve():
    x = np.linspace(0, 1, 1000)

    plt.plot(x, f(x), color='red')

    plt.grid(True)
    plt.title("IoU-to-Area Curve")
    plt.xlabel("IoU Value")
    plt.ylabel("Overlapped Area")

    plt.show()

if __name__ == "__main__":
    plot_iou_curve()


