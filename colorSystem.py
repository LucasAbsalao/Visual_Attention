import numpy as np
import cv2


def multiply_matrix(transformation_matrix, vector):
    return np.dot(transformation_matrix, vector)
    

def rgb2xyz(img):
    transformation_matrix = [[0.4124, 0.3576, 0.1805],
                             [0.2126, 0.7152, 0.0722],
                             [0.0193, 0.1192, 0.9505]]
    new_img = np.zeros(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = multiply_matrix(transformation_matrix,img[i][j])
            
    return new_img

def f_lab(epsilon, k, t):
    result = 0
    if t > epsilon:
        result = t**(1/3)
    else:
        result = ((k*t) + 16)/116
    return result

def rgb2lab(img):
    img_xyz = rgb2xyz(img)
    print(img_xyz)

    Xn = 95.0489
    Yn = 100
    Zn = 108.884

    e = 0.008856
    k = 903.3

    img_Lab = np.zeros(img_xyz.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_Lab[i][j][0] = 116* f_lab(e,k,img_xyz[i][j][1]/Yn) - 16
            img_Lab[i][j][1] = 500*(f_lab(e,k,img_xyz[i][j][0]/Xn) - f_lab(e,k,img_xyz[i][j][1]/Yn))
            img_Lab[i][j][2] = 200*(f_lab(e,k,img_xyz[i][j][1]/Yn) - f_lab(e,k,img_xyz[i][j][2]/Zn))

    return img_Lab#np.uint8(np.round(img_Lab))


matrix = np.array([[[1,2,3], [3,2,6], [9,8,4]],
                   [[4,5,6], [2,6,4], [4,5,8]],
                   [[3,8,7], [8,4,3], [3,7,9]]])

print(rgb2lab(matrix))