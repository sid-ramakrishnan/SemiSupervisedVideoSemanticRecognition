import numpy as np
import math
from dataFile import input_matrix


np.random.seed(1)



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



def getNeighbors(input_matrix,index,k):
    x = input_matrix[index]
    neighbourlist = []
    distances = []    
    indices   = []    
    for i in range(len(input_matrix)):
        y = input_matrix[i]
        dist = 0
                
        for j in range(len(x[0])):
                dist += (x[0][j] - y[0][j])*(x[0][j] - y[0][j])
        distances.append(math.sqrt(dist))
        indices.append(i)
    distance_pairs = zip(distances, indices)
    
    distance_pairs.sort(key = lambda x:x[0])
    
    for count in range(k):
        neighbourlist.append(distance_pairs[count][1])
        
         
    return neighbourlist
    
def greenFunction(x1,x2,e):
    couloumbConstant = float(1)/float(4*math.pi*e)
    dist = 0    
    #print x1
    #print x2    
    for j in range(len(x1[0])):
                #print x1[0,j]
                #print x2[0,j]
                dist += (x1[0,j] - x2[0,j])*(x1[0,j] - x2[0,j])
    dist = couloumbConstant * math.sqrt(dist)
    return dist
     





m = mean( input_matrix , 34 , 28 , 9 )
inputMatrix = m[0:12,0:9]

labelledscatterMatrix = np.zeros((9,9))
meanlabelledlist = []
for i in range(inputMatrix.shape[1]):
   meanlabelledlist.append(np.mean(inputMatrix[:,i])) 



mean_vector = np.array([meanlabelledlist])



for i in range(inputMatrix.shape[1]):
   labelledscatterMatrix += (inputMatrix[i,:] - mean_vector).dot(
                    (inputMatrix[i,:] - mean_vector).T)
#print('Scatter Matrix:\n', labelledscatterMatrix)

#Scatter Matrix for labelled instances is complete

#Now we need to take the entire matrix and find local cliques for each data
#point x. create a list with the indices of the k-1 nearest neighbours of x 
#and add x to this list

#first we assign a value to k
k = 5
localCliqueMatrix = np.zeros((34,k))
KMatrix = np.zeros((34,k,k))
    

for i in range(localCliqueMatrix.shape[0]):
  neighbours = getNeighbors(input_matrix,i,5)
  #print neighbours
  localCliqueMatrix[i] = neighbours 
  
  #For these neighbours we need to construct a k*k matrix Ki which is Green's
  #function of x1,x2. We use Coulomb's green function where episilon = 4
  for j in range(len(neighbours)):
      for k in range(len(neighbours)):
          KMatrix[i][j][k] = greenFunction(m[neighbours[j]],m[neighbours[k]],4)
  #print ("index",i)  
  #print KMatrix[i]   

#PCA











