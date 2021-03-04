import numpy as np
from matplotlib import pyplot as plt
import math
import random

myImage = np.zeros((240,320))


PI = 3.14159
segments = 600
spacing = PI*2 / segments


def addSquare():
    pass

def addCircle(pX, pY, size):
    for i in range(segments):
        for j in range(size):
            x = math.cos(spacing * i) * (size-j)
            y = math.sin(spacing * i) * (size-j)
            myImage[math.floor(x) + pX, math.floor(y) + pY] = not myImage[math.floor(x) + pX, math.floor(y) + pY]
if __name__ == "__main__":
    #
    for i in range(100):
        x = random.randint(0, 220)
        y = random.randint(0, 300)
        size = random.randint(0, 20)
        addCircle(x,y,size)
    plt.imshow(myImage, interpolation="bilinear", clim=(0,1),cmap="gray")
    plt.show()
