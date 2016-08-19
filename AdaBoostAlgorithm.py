from __future__ import division
from numpy import *

#AdaBoost Algorithm
#let us assume that training set is empty intially.

training_set = []
n_training_set = 0 #n_training_set represents the length of the training set
weights = [] 

'''
In AdaBoost Algorithm the initial weights are assumed to be 1
for each of the element in the training_set.
And since we want it as a distribution, we normalize it by
dividing it with total sum of all the weights.

Also, AdaBoost algorithm builds a classifier on top of many 
weak classifiers so that finally we have a boosted 
classifier algorithm.

Remember in the AdaBoost Algorithm, the classifier with
accuracy less than 50 percent are given a negative values and 
with 50 percent accuracy a zero and more than fifty percent 
a positive value is assigned for the classifier. We are talking
about the weak classifiers here. 

RULES_CLASSIFIERS is the variable that holds the function that is 
the various weak classifiers in it.

ALPHA_CLASSIFIERS is the variable that is gonna store the score
for each of the weak classifiers.
'''

def rule_classifier( func_classifier ) :
    global weights
    errors_classifier = [ ]
    
    for training_example in training_set :
    
        if training_example[ 1 ] != func_classifier( training_example[ 0 ] ) :
            errors_classifier.append( 1 )
        else :
            errors_classifier.append( 0 )

    e_classifier = 0.000001
    for i in xrange( n_training_set ) :
        e_classifier = e_classifier + errors_classifier[ i ] * weights[ i ]  

    log_value_classifier = log( ( 1 - e_classifier ) / e_classifier )
    
    
    alpha_classifier = 0.5 * log_value_classifier

    w_classifier = []

    for i in xrange( n_training_set ) :
        w_classifier.append( 0 )

    for i in xrange( n_training_set ) :
        if errors_classifier[ i ] == 1 :
            w_classifier[ i ] = weights[ i ] * exp( alpha_classifier )
        else :
            w_classifier[ i ] = weights[ i ] * exp( -alpha_classifier )
        
    weights = w_classifier / sum( w_classifier )
        
    RULES_CLASSIFIERS.append( func_classifier )
    ALPHA_CLASSIFIERS.append( alpha_classifier )
        

def evaluating_classifiers( ) :

    n_classifiers = len( RULES_CLASSIFIERS )

    for ( x_parameter , check ) in training_set :

        final_adaboost_classifier = [ ALPHA_CLASSIFIERS[ i ] * RULES_CLASSIFIERS[ i ]( x_parameter ) for i in xrange( n_classifiers ) ]
        print x_parameter, sign( check ) == sign( sum( final_adaboost_classifier ) )







RULES_CLASSIFIERS = [] 
ALPHA_CLASSIFIERS = []

if __name__ == '__main__' :
    training_examples = []
    training_examples.append(((1,  2  ), 1))
    training_examples.append(((1,  4  ), 1))
    training_examples.append(((2.5,5.5), 1))
    training_examples.append(((3.5,6.5), 1))
    training_examples.append(((4,  5.4), 1))
    training_examples.append(((2,  1  ),-1))
    training_examples.append(((2,  4  ),-1))
    training_examples.append(((3.5,3.5),-1))
    training_examples.append(((5,  2  ),-1))
    training_examples.append(((5,  5.5),-1))

    n_training_set = len( training_set )
    for i in xrange( n_training_set ) :
        weights.append( 1 / n_training_set )
    training_set = training_examples
    
    rule_classifier(lambda x: 2*(x[0] < 1.5)-1)
    rule_classifier(lambda x: 2*(x[0] < 4.5)-1)
    rule_classifier(lambda x: 2*(x[1] > 5)-1)
    evaluating_classifiers( )
    
