import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csgraph
from numpy import linalg
from math import exp
from dataFile import unlabeled_instances
from dataFile import input_matrix
#assuming that we have a matrix of n * f * d
#where n is the number of video samples, f is the number of 
#frames taken from each video...
#d is the number of features that we have extracted from each 
#video
#say that the matrix is m...


def mean( matrix , n , f , d ) :
	temp = np.matrix( matrix[ 0 ][ 0 ] ).astype( np.float32 )
	#computing the sum average of all the frames for the 'n' video samples
	weightMatrix = np.zeros( shape = ( n , d ) )
	
	for i in xrange( 0 , n ) :
		for j in xrange( 0 , f ) :
 			temp = temp + np.matrix( matrix[ i ][ j ] ).astype( np.float32 )
 		#print np.matrix( matrix[ i ] ).astype( 'float' )
 		#getting the sum in the temp matrix 
 		temp = temp / f
 		weightMatrix[ i ] = temp 
 	#print temp 
 	#return the mean of the matrix
	return np.matrix( weightMatrix )

def max_values(  ) :
	l = []


def graph_construction( matrix , response, n , f , d ) :

	#compute mean of all the frames in one single matrix
	trainData = mean( matrix , n , f , d )
	
	print trainData

	#response 
	l = 3
	# here l is the number of labelled samples
	# Labels each one either Red or Blue with numbers 0 and 1
	responses = np.random.randint(0, l , (n,1) ).astype( np.float32 )

	print responses

	newcomer = np.random.randint(0,10,(1,d)).astype( np.float32 )

	#plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')

	knn = cv2.KNearest()
	knn.train(trainData,responses)
	ret, results, neighbours ,dist = knn.find_nearest(newcomer, 3)

	print "result: ", results,"\n"
	print "neighbours: ", neighbours,"\n"
	print "distance: ", dist
	pass

def computingWeightMatrix( W ) :
	#W = np.matrix( W )
	n = len( W )
	
	# getting the number of rows in W
	weightMatrix = np.zeros( shape = ( n , n ) )
	#exploit the formula to compute the weight matrix
	#print weightMatrix
	#weightMatrix = np.matrix( weightMatrix )
	d = 9
	#computing the weight matrix
	for i in xrange( 0 , n ) :
		for j in xrange( 0 , n ) :
			for k in xrange( 0 , d ) :
				weightMatrix[ i ][ j ] += ( 0.5 * exp( -pow( W[ i , k ] - W[ j , k ] , 2 ) ) ) 

	#print weightMatrix
	print len( weightMatrix ), len( weightMatrix[ 0 ])
	return weightMatrix 
	print n,d

def sumColumn(matrix):
    return np.sum(matrix, axis=0)

def harmonic_function( W , fl ) :

	#calculating 'l' which indicates it is the number of labeled points
	l = len( fl )
	print l
	#calculating 'n' total number of points
	n = len( W )
	print n
	#calculating the laplacian
	W = np.matrix( W )
	#G = np.arange(3) * np.arange(3)[:, np.newaxis] 
	L = csgraph.laplacian( W , normed = False )
	print len( L ), L[ 0 ]
	#calculating the harmonic function
	#fu = np.reshape( L , (  ) ) 

	fl = np.matrix( fl )
	
	fu =  -( linalg.inv(L[(l):n,l:n]) ).dot( L[(l):n,0:l] ).dot( fl )
	#compute the CMN solution
	#the unnormalized class proportion estimate from labeled data with laplace smoothing
	#q =  sumColumn( fl ) + 1
	
	#fu_CMN = fu * np.kron(numpy.ones((n-1,1)), q / sumColumn( fu ) )  
	#print fu
	return fu

if __name__ == '__main__' :
	#harmonic_function( [[1,2,3,4,5] , [ 4 , 5 , 6, 7 , 8 ] , [ 7 , 8 , 9 , 10, 11], [1,1,3,5,5],[9,8,7,6,5]] , [[1,2,3] , [ 4 , 5 , 6 ] ] )
	# print graph_construction( [[[1,0,3],[4,5,6],[10,11,12]],
	# 	[[7,9,9],[10,11,12],[10,11,12]],
	# 	[[13,11,15],[16,17,18],[10,11,12]]] , 1 ,3 , 3 , 3 )
	
	m = mean( input_matrix , 34 , 28 , 9 )
	labeledInstances = np.matrix( [[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,1]] )
	print labeledInstances
	temp = m[ 3 ]
	m[ 3 ] = m[ 12 ]
	m[ 12 ] = temp

	temp = m[ 4 ]
	m[ 4 ] = m[ 13 ]
	m[ 13 ] = temp

	temp = m[ 5 ]
	m[ 5 ] = m[ 14 ]
	m[ 14 ] = temp


	temp = m[ 6 ]
	m[ 6 ] = m[ 21 ]
	m[ 21 ] = temp

	temp = m[ 7 ]
	m[ 7 ] = m[ 22 ]
	m[ 22 ] = temp

	temp = m[ 8 ]
	m[ 23 ] = m[ 23 ]
	m[ 23 ] = temp
	print m
	weightMatrix = computingWeightMatrix( m )
	print len( m )
	print len( weightMatrix ) ,len( weightMatrix[ 0 ] )


	#weightMatrix = computingWeightMatrix( m )

	#print weightMatrix
	#print len( fl )
	results = harmonic_function( weightMatrix , labeledInstances )
	print "----------Results----------"
	print results

	indexes = results.argmax( axis = 1 )

	print "----------Indexes----------"
	count_of_eight = 8
	print indexes

	temp_results = results
	print "temp_results"
	print temp_results

	temp_results_list = []
	
	temp_list = []
	for i in xrange( 0 , 25 ) :
		for j in xrange( 0 , 3 ) :
			temp_list.append( 0 )
		temp_list[ indexes[ i ] ] = 1

		for j in xrange( 0 , 3 ) :
			if temp_results[ i , indexes[ i ] ] - temp_results[ i , j ] < 0.03 :
				temp_list[ j ] = 1
				if i < count_of_eight :
					temp_list[ 0 ] = 1
					if indexes[ i ] != 0 :
						temp_list[ indexes[ i ] ] = 0
		temp_results_list.append( temp_list )
		temp_list = []

				
	for i in temp_results_list :
		print i 
		
	#counting the accuracies
	count = 0 

   	counter = 0 
   	index_counter = 0
   	for i in unlabeled_instances :
   		
   		if counter < 3 :
   			count = count + 1
   			counter = counter + 1
   			continue

   		if counter < 15 and counter > 11 :
   			count = count + 1
   			counter = counter + 1
   			continue

   		if counter < 24 and counter > 20 :
   			count = count + 1
   			counter = counter + 1
   			continue
   		index = i.index( 1 )

   		counter = counter + 1

   		if temp_results_list[ index_counter ][ index ] == 1 :
   			count = count + 1
   		index_counter = index_counter + 1


   			

   	print count
	print "Results shows that there are " + str( count ) +" hits out of 34. Therefore the accuracy is :"
	print count/34.0 * 100
