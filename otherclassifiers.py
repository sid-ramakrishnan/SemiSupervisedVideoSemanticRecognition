# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:59:58 2016

@author: macbook
"""
import numpy as np
import os
from scattermatrix import mean
from sklearn.cluster import KMeans
os.chdir("/Users/macbook/Downloads")
c = np.loadtxt("testinput.txt")
feature_matrix = c.reshape(181,43,9)
m = mean(feature_matrix , 181 , 43 , 9 )
kmeans = KMeans(n_clusters=4)
kmeans.fit(m)


