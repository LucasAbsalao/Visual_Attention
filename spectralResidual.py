import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import os
from show import showAndDestroy, plot_img



PATH_IMAGENS = 'images/'

img = cv2.imread(PATH_IMAGENS + 'flower2.webp')

showAndDestroy("teste", img)