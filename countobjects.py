# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:49:57 2015

@author: p
"""
import numpy as np
import cv2
import os

def countObjects(img):
 pathforkeyframeswalk = "/home/p/Desktop/Major-Project/keyFramesWalk/"
 os.chdir(pathforkeyframeswalk)    
 im = cv2.imread(img,0)
 img_filt = cv2.medianBlur(im, 5)
 img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
 contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
 print len(contours[0])
 return len(contours[0])
