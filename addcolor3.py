# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 18:27:04 2021

@author: hanso
"""

import cv2
import numpy as np
from pylab import *
from PIL import Image
from gradient import forward_differences, forward_differences_conj, prox_project


from numpy import *
 
def weighed1(im, clambda, gt, iter=1000):
 
    #初始化
    U = im
    U0 = im
    h=im.shape[1]
    i=0
    '''
    while(i<=iter):
        U=clambda*((gt/255)*((roll(U,1,axis=1)-U)+(roll(U,-1,axis=1)-U)+(roll(U,1,axis=0)-U)+(roll(U,-1,axis=0)-U))+2*(gx/255)*((roll(U,1,axis=0)-U)+(roll(U,-1,axis=0)-U))+2*(gy/255)*((roll(U,1,axis=1)-U)+(roll(U,-1,axis=1)-U)))/4 +U
        i+=1
        if (i%1000==0):
            print(i);
    '''
    while(i<=iter):
        Cx = roll(U,1,axis=0)-U
        Cy = roll(U,1,axis=1)-U
        Cxx = roll(U,1,axis=0)+roll(U,-1,axis=0)-2*U
        Cyy = roll(U,1,axis=1)+roll(U,-1,axis=1)-2*U
        gx = roll(gt, 1, axis = 0) - gt
        gy = roll(gt, 1, axis = 1) - gt
        U = U +0.0005*(gx*Cx + gy*Cy + gt*Cxx + gt*Cyy - clambda * (U - U0))
        i+=1
    return U
