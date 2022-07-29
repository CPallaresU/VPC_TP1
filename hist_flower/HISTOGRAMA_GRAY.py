# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 16:48:35 2022

@author: 10
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def hist_(path):

    img = cv.imread(path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([img_gray],[0],None,[256],[0,256])

    # compute and plot the image histograms
    # [max value to show] [number of bins] for cv.calHist

    plt.plot(hist)
        
    plt.title('Image Histogram GFG')
    plt.show()
    
    
hist_("img1_tp.png")
hist_("img2_tp.png")

img1 = cv.imread("img1_tp.png")
img2 = cv.imread("img2_tp.png")
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

vec1 = np.sort(np.reshape(img1_gray, 288*287))
vec2 = np.sort(np.reshape(img1_gray, 288*287))

img1_gray.sort()
img2_gray.sort()

cont = 0

for t in range(len(vec1)):
    if vec1[t] == vec2[t]:
        cont = cont +1