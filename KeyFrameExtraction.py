import numpy as np
import cv2
import os
import sys
import csv
from os import listdir
from os.path import isfile, join
'''
Steps to be followed
1 ) Pre process each of the images and store the preprocessed images in the destination folder preprocessedimages
2 ) Extract keyframes for each image and store them in the walking keyframesfolder  


'''
imageList = csv.reader((open("imagelist.csv","rb")))
print imageList
for row in imageList:
    print row[0]
    os.system("python preprocess.py --command shrink --source '"+row[0]+"' --dest '/home/p/Desktop/Major-Project/preprocessedimages/"+row[0]+"'")

#now reduced videos of under a second have been created. Next to extract keyframes

pathforreducedimage = "/home/p/Desktop/Major-Project/preprocessedimages" 
reducedImageFiles = [ f for f in listdir(pathforreducedimage) if isfile(join(pathforreducedimage,f)) ]
pathforkeyframeswalk = "/home/p/Desktop/Major-Project/keyFramesWalk/"
os.chdir(pathforkeyframeswalk)
for file in reducedImageFiles:
    cap = cv2.VideoCapture(pathforreducedimage+"/"+file)
    ret = True
    count = 0
    while(ret):
       ret, frame = cap.read()

       if(ret==False):
         break  
       else:
         #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         cv2.imwrite("Image"+file+" "+str(count)+".png",frame)
         count = count+1      
         if cv2.waitKey(25) & 0xFF == ord('q'):
          break

    cap.release()
    cv2.destroyAllWindows()