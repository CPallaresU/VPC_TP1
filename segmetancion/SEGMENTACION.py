# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 20:56:59 2022

@author: 10
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



def grey_eqz(path):
    

    fig = plt.figure()
    img = cv.imread('path',cv.IMREAD_GRAYSCALE)
    
    
    # Imagen original
    ax1=plt.subplot(221)
    ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Original')
    
    hist1,bins1 = np.histogram(img.ravel(),256,[0,256])
    ax3=plt.subplot(223)
    ax3.plot(hist1)
    
    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eqzd = clahe.apply(img)
    ax2=plt.subplot(222)
    ax2.imshow(img_eqzd, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('Ecualizada')
    
    hist2,bins2 = np.histogram(img_eqzd.ravel(),256,[0,256])
    ax4=plt.subplot(224)
    ax4.plot(hist2)
    
    fig.show()
    
    # Mostrar las imagenes lado a lado usando cv2.hconcat
    out1 = cv.hconcat([img, img_eqzd])
    cv.imshow('a',out1)
    t = cv.cvtColor(img_eqzd,cv.COLOR_GRAY2RGB)
    cv.imwrite("eqz_img.jpg",t)


def color_eqz(image_path):
    rgb_img = cv.imread(image_path)

    # convert from RGB color-space to YCrCb
    ycrcb_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)

    cv.imwrite("equalized_img.jpg",equalized_img)
    
    return equalized_img




def segmentation_(path):
    
    img = cv.imread(path)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hist = cv.calcHist([img],[1],None,[256],[0,256])
    
    colors = ('b','g','r')
    #b,g,r
    #0,1,2
      
    # compute and plot the image histograms
    # [max value to show] [number of bins] for cv.calHist
    
    for i,color in enumerate(colors):
        hist = cv.calcHist([img],[i],None,[256],[0,100])
        plt.plot(hist,color = color)
        
    plt.title('Image Histogram GFG')
    plt.show()
    
    
    
    # Cargar la imagen color
    #-----------------------
    img_color = cv.imread(path)
    imgRGB = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)
    img_HSV = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
    
    #ROCAS
    color_l = (0,0,10) 
    color_u = (50,255,255)
    
    mask_roca = cv.inRange(img_HSV, color_l,  color_u)
    img_segmentada = cv.bitwise_and(imgRGB, imgRGB, mask=mask_roca)
    
    cv.imwrite("image_ROCAS1.jpg",img_segmentada)
    
    
    #MAR
    
    #color_l = (0,100,50) 
    #color_u = (255,255,100)
    color_l = (50,50,50) 
    color_u = (255,255,150)
    
    mask_mar = cv.inRange(img_HSV, color_l,  color_u)
    img_segmentada = cv.bitwise_or(imgRGB, imgRGB, mask=mask_mar)
    
    cv.imwrite("image_MAR1.jpg",img_segmentada)
    
    
    #CIELO
    color_l = (50,50,200) 
    color_u = (255,255,255)
    
    mask_cielo = cv.inRange(img_HSV, color_l,  color_u)
    img_segmentada = cv.bitwise_or(imgRGB, imgRGB, mask=mask_cielo)
    
    cv.imwrite("image_CIELO1.jpg",img_segmentada)
    
    return mask_roca, mask_mar, mask_cielo



    
eqz_img = color_eqz('segmentacion.jpg')
masks = segmentation_('equalized_img.jpg')


B, G, R = cv.split(cv.imread('segmentacion.jpg'))

B = np.reshape(B, 598484)
G = np.reshape(G, 598484)
R = np.reshape(R, 598484)



imgs=cv.imread("equalized_img.jpg")
cielo=cv.imread("image_CIELO1.jpg")
mar=cv.imread("image_MAR1.jpg")
rocas=cv.imread("image_ROCAS1.jpg")

#subs = rocas - mar
cv.imwrite("img-rocas.jpg", imgs - rocas)
cv.imwrite("img-mar.jpg", imgs - mar)
cv.imwrite("img-cielo.jpg", imgs - cielo)


#plt.scatter(list(B[:500000]),list(G[:500000]),alpha=0.01)




"""

rocas = cv.imread('image_ROCAS1.jpg',0)*255
mar = cv.imread('image_MAR1.jpg',0)*255

for i in range(np.shape(rocas)[0]):
    for j in range(np.shape(rocas)[1]):
        
        if rocas[i][j]>0:
            rocas[i][j] = 255
        else:
            rocas[i][j] = 0
            
        if mar[i][j]>0:
            mar[i][j] = 255
        else:
            mar[i][j] = 0




"""
