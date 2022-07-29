# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 19:20:20 2022

@author: 10
"""

import cv2 as cv
import numpy as np

## 9138678373
## https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero aporte con 50 puntos

def div_by_zero (numerator,denominator):

    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(numerator,denominator)
        c[c == np.inf] = 0
        c = np.nan_to_num(c)
    
    return c


#WHITE-PATCH Y COORDENADAS CROMATICAS

B, G, R = cv.split(cv.imread('wp_green3.jpg'))

B = B.astype(np.uint32) * div_by_zero([255],[np.amax(B)])[0]
G = G.astype(np.uint32)* div_by_zero([255],[np.amax(G)])[0]
R = R.astype(np.uint32)* div_by_zero([255],[np.amax(R)])[0]

merged = cv.merge([B, G, R])

cv.imwrite("wp_patch_green.png", merged.astype(np.uint8))