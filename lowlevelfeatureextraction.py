# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:45:35 2015
@author: p
"""
import numpy as np
import cv2
import os

from shannonsEntropy import calcentropy
from whitecontrib import whiteContrib
from countobjects import countObjects
'''compute the first two color moments of an image(which has colour),grayscale images are represented bas
% input: image to be analyzed and extract 2 first moments from each R,G,B
% output: 1x6 vector containing the 2 first color momenst from each  R,G,B


% construct output vector
colorMoments = zeros(1, 6);
colorMoments(1, :) = [meanR stdR meanG stdG meanB stdB];

% clear workspace
clear('R', 'G', 'B', 'meanR', 'stdR', 'meanG', 'stdG', 'meanB', 'stdB');

end'''

'''Steps to follow
1) Go into the keyframeswalk directory
2) for each image calculate all params
3) Store in a lowlevelfeature matrix for each image
   dimensions
   1-Which Main video we are considering
   2-keyframe number corresponding to the main video
   3-color moments 
'''

'''
Calculate the Shannon Entropy for a given image
This is dimension number 7 (6 in terms of array indexing in our feature matrix)

Next we calculate how much of the image is dominated by white components which
is stored in dimension 8

We calculate the number of contours present in the image, this is a slightly
inaccurate measure for object detection due to the presence of noise but for
now,we keep it - dimension 9
'''

'''
We label the first 3 training examples of every type of video processed by 
creating a matrix of N videos * d categories 
0-Human Being
1-Nature


'''

lowlvlfeatures = np.zeros((34,28,9),float)
labelmatrix = np.zeros((34,3),int)
pathforkeyframeswalk = "/home/p/Desktop/Major-Project/keyFramesWalk/"
os.chdir(pathforkeyframeswalk)
workingdirectory = os.listdir( pathforkeyframeswalk )

# This would print all the files and directories
for file in workingdirectory:
      
   split = file.split('.')
   firstpart = split[0].split(' ')
   videonum = int(firstpart[1])
   
   secondpart = split[1].split(' ')
   imagenum = int(secondpart[1])
   
   img = cv2.imread(file)   
   B = img[:,:,0]
   G = img[:,:,1]
   R = img[:,:,2]
   meanR = np.mean(R[:])
   stdR = np.std(R[:])
   meanG = np.mean(G[:])
   stdG = np.std(G[:])
   meanB = np.mean(B[:])
   stdB = np.std(B[:])
   
  
   
   entropy = calcentropy(file,B,G,R)
   whitecontrib = whiteContrib(file)
   countobj = countObjects(file)
       
   
   lowlvlfeatures[videonum,imagenum,:] = [meanR, stdR, meanG, stdG, meanB, stdB,entropy,whitecontrib,countobj]
      
#Labelling the Human videos
labelmatrix[0] = [1,0,0]
labelmatrix[1] = [1,0,0]
labelmatrix[2] = [1,0,0]

labelmatrix[12] = [0,1,0]
labelmatrix[13] = [0,1,0]
labelmatrix[14] = [0,1,0]
   
labelmatrix[21] = [0,0,1]
labelmatrix[22] = [0,0,1]
labelmatrix[23] = [0,0,1]
   
