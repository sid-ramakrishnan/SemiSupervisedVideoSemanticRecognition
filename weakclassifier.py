# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 23:47:32 2016

@author: Rohit Kumar

1 -> Human
0 ->Building
-1 -> Nature
"""

from dataFile import unlabeled_instances
import numpy as np
import math

n_training_set = 34

true_value = []
for i in range(len(unlabeled_instances)):
    if(unlabeled_instances[i][0] == 1):
        true_value.append(1)
    elif(unlabeled_instances[i][1] == 1):
        true_value.append(0)
    if(unlabeled_instances[i][2] == 1):
        true_value.append(-1)    
print ("True value of Y",true_value)

errors_classifier = [1,1,1,1,0,1,-1,-1,1,1,1,1,0,0,0,0,-1,1,1,-1,1,0,1,-1,-1,-1,-1,-1,0,0,0,1,1,1,-1,-1]
#Error classifier is the value that our algorithm has returned. (Weak classifier)

#weights score 0.1 if there is a correct match
#-0.05 if there is an incorrect match

weights = []
for i in range(len(true_value)):
    if (true_value[i]==errors_classifier[i]):
        weights.append(0.1)
    else :
        weights.append(-0.05)

print ("Weights list",weights)
e_classifier = 0.000001
for i in xrange( n_training_set ) :
        e_classifier = e_classifier + errors_classifier[ i ] * weights[ i ]  


print ("E_Classifier",e_classifier)
log_value_classifier = math.log( ( 1 - e_classifier ) / e_classifier )
    
    
alpha_classifier = 0.5 * log_value_classifier

print ("Alpha classifier",alpha_classifier)