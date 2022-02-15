# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:27:06 2021

@author: hanso
"""

import cv2
from pylab import *
from PIL import Image
from addcolor3 import weighed1
import numpy as np
from rofdenoise import denoisel1

img = cv2.imread("gradient2.png",cv2.IMREAD_GRAYSCALE)

sobelX = cv2.Sobel (img,cv2.CV_64F,1,0)
sobelY = cv2.Sobel(img,cv2.CV_64F,0,1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX,sobelY)
sobel=np.array(sobelCombined)
sobelX   = denoisel1(sobelX,20, 0.125, 800)
sobelY   = denoisel1(sobelY,20, 0.125, 800)
sobel   = denoisel1(sobel,20, 0.125, 800)

gt = np.zeros((sobel.shape[0],sobel.shape[1]))
gx = np.zeros((sobel.shape[0],sobel.shape[1]))
gy = np.zeros((sobel.shape[0],sobel.shape[1]))

for i in range(sobel.shape[0]):
    for j in range(sobel.shape[1]):
        gt[i,j]=(255-sobel[i,j])

        if(i<10 or j<10 or i>=sobel.shape[1]-10 or j>=sobel.shape[1]-10):
            gt[i,j]=0


        if (gt[i,j]>= 200):
            gt[i,j] = 250
        if (gt[i,j]<=150):
            gt[i,j] = 0

img = cv2.imread('uncolored13.png')
original = cv2.imread('uncolored13.png')
img = img.astype(np.float32)

b,g,r = cv2.split(img)
orb,org,orr = cv2.split(original)

arr=np.zeros((b.shape[0],b.shape[1]))
for i in range (0,arr.shape[0]):
    for j in range (0,arr.shape[1]):
        if ((30<=i<=110 and 30<=j<=110)or(150<=i<=230 and 150<=j<=230)):
            arr[i,j]=10
            '''
        if (40>i>20 and 20<j<40):
            arr[i,j]=10
            '''
        else:
            arr[i,j]=0

h=1/img.shape[0]
iters=100000
brightness = np.zeros((arr.shape[0],arr.shape[1]))
originalb = np.zeros((arr.shape[0],arr.shape[1]))

for i in range (0,arr.shape[0]):
    for j in range (0,arr.shape[1]):
        brightness[i,j] = sqrt (b[i,j]**2+g[i,j]**2+r[i,j]**2)
        originalb[i,j] = sqrt (orb[i,j]**2+ org[i,j]**2 +orr[i,j]**2)
        b[i,j]/=brightness[i,j]
        r[i,j]/=brightness[i,j]
        g[i,j]/=brightness[i,j]
        
for i in range (0,arr.shape[0]):
    for j in range (0,arr.shape[1]):
        if(i<=5 or j<=5 or i>=sobel.shape[1]-5 or j>=sobel.shape[1]-5):
            b[i,j]=b[6,6]
            g[i,j]=g[6,6]
            r[i,j]=r[6,6]
            
b_colored = weighed1(b,arr, gt, iters)
g_colored = weighed1(g,arr, gt, iters)
r_colored = weighed1(r,arr, gt, iters)

chromaticity_colored = cv2.merge([b_colored, g_colored, r_colored])

for i in range (0,arr.shape[0]):
    for j in range (0,arr.shape[1]):
        b_colored[i,j]*=originalb[i,j]
        r_colored[i,j]*=originalb[i,j]
        g_colored[i,j]*=originalb[i,j]


colored = cv2.merge([b_colored,g_colored,r_colored])

gt = np.clip(gt, 0, 255).astype(np.uint8)
img = np.clip(img, 0, 255).astype(np.uint8)
colored   = np.clip(colored, 0, 255).astype(np.uint8)
r_colored = np.clip(r_colored, 0, 255).astype(np.uint8)
cv2.imshow('gradient'  ,  gt)
cv2.imshow('chrom_colored'  ,  chromaticity_colored )
cv2.imshow('uncolored'  , img  )
cv2.imshow('colored'  , colored  )
cv2.waitKey(0)
cv2.destroyAllWindows()
