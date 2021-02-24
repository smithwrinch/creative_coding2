import numpy as np
import cv2

def decimalToBinary(n):
    return bin(n).replace("0b", "")

img = cv2.imread('python_challenge.png')
r = 0
g = 0
b = 0
print(img.shape)

for row in range(img.shape[0]):
    for col in range(img.shape[1]):
            rr = img[row][col][0]
            bb = img[row][col][1]
            gg = img[row][col][2]
            if(rr + bb + gg == 255):
                if(rr > 0):
                    r += rr
                elif(gg > 0):
                    g += gg
                else:
                    b += bb

r /= 255
g /= 255
b /= 255

print(r)
print(g)
print(b)

number = decimalToBinary(int(r)) + decimalToBinary(int(g)) + decimalToBinary(int(b))
print(number)

number_converted = int(number, 2)
print(number_converted)
