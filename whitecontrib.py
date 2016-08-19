# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 00:38:26 2015
@author: p
"""
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)

def whiteContrib(img):
 pathforkeyframeswalk = "/home/p/Desktop/Major-Project/keyFramesWalk/"
 os.chdir(pathforkeyframeswalk)
 img = cv2.imread(img, 0)
 ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 flattenedarray = thresh.flatten()
 no_of_pixels = thresh.size
 contribwhite = float(list(flattenedarray).count(255))/no_of_pixels
 return contribwhite
