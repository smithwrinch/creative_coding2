import numpy as np
from matplotlib import pyplot as plt
import math

myImage = np.zeros((240,320))


PI = 3.14159

if __name__ == "__main__":

    for i in range(240):
        for j in range(320):
            # t = math.tan(i / frequency) * math.cos(j / frequency) + math.atan(j / frequency) * math.cos(i / frequency);
            myImage[i,j]= 1000

    plt.imshow(myImage)
    plt.show()
