# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 22:48:46 2022

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

B, G, R = cv.split(cv.imread('CoordCrom_3.png'))

B = B.astype(np.uint32)
G = G.astype(np.uint32)
R = R.astype(np.uint32)

denominator_ = (B+R+G)
denominator = np.reshape(denominator_, np.shape(B)[0]*np.shape(B)[1])


B_ = np.reshape(B, np.shape(B)[0]*np.shape(B)[1])
G_ = np.reshape(G, np.shape(B)[0]*np.shape(B)[1])
R_ = np.reshape(R, np.shape(B)[0]*np.shape(B)[1])


B_cc = np.reshape(div_by_zero(B_,denominator) , (np.shape(B)[0],np.shape(B)[1]))
G_cc = np.reshape(div_by_zero(G_,denominator) , (np.shape(B)[0],np.shape(B)[1]))
R_cc = np.reshape(div_by_zero(R_,denominator) , (np.shape(B)[0],np.shape(B)[1]))

merged = cv.merge([B_cc*255, G_cc*255, R_cc*255])

cv.imwrite("CoordCrom_3_coor.png", merged)

    
"""
B, G, R = cv.split(cv.imread('CoordCrom_2.png'))
merged_new = cv.merge([B-10, G-10, R-10])
cv.imwrite("CoordCrom_2_2.png", merged_new)
"""

