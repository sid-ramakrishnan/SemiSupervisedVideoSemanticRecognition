# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 00:33:27 2015
@author: p
"""

import numpy as np
import cv2
import os
import sys
import math

def getfrequencyofbin(mat,hist):
   
   frequency = 0.0;
   for i in range(len(hist)):
       Hc = hist[i]
       frequency += Hc;
   return frequency;   
   
   
   
def calcentropy(img,B,G,R):

 entropy = 0.0
 img = cv2.imread(img)
 histB = cv2.calcHist([B],[0],None,[256],[0,256])
 histG = cv2.calcHist([G],[0],None,[256],[0,256])
 histR = cv2.calcHist([R],[0],None,[256],[0,256])
 normB = [float(i)/sum(histB) for i in histB]
 normG = [float(i)/sum(histG) for i in histG]
 normR = [float(i)/sum(histR) for i in histR]

 frequencyR = getfrequencyofbin(R,normR)
 for i in range(len(normR)):
    Hc = normR[i]
    if(Hc!=0.):    
      entropy += -(Hc/frequencyR) * math.log10((Hc/frequencyR));
     
 frequencyG = getfrequencyofbin(G,normG)
 for i in range(len(normG)):
    Hc = normG[i]
    if(Hc!=0.):    
     entropy += -(Hc/frequencyG) * math.log10((Hc/frequencyG));

 frequencyB = getfrequencyofbin(B,normB)
 for i in range(len(normB)):
    Hc = normB[i]
    if(Hc!=0.):    
     entropy += -(Hc/frequencyB) * math.log10((Hc/frequencyB));


 return entropy