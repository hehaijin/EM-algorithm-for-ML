
import numpy as np
from Mahalanobis  import Mahalanobis


def Kmeans(data,mus,coMs):
	N=data.shape[0]  #number of samples
	K=mus.shape[0]  # number of classes
	ri=np.zeros(K) #similar to GMM, but here we will use hard EM
	result=Mahalanobis(D,mus,coMs)
	for i range(N):
		maxi=result[i][0];
		zi=0
		for j in range(K):
			if result[i][j] > maxi:
				maxi=result[i][j]
				zi=j
		ri[i]=zi
	
	return ri
	   
def KmeansUpdate(data,ri,K):
	N=data.shape[0]
	D=data.shape[1]
    rmus=np.zeros((N,K))
    rcoMs=np.zeros((K,D,D))
    #create EM style rik so to reuse code
    rik=np.zeros((N,K))
    rk=numpy.zeros(K)
    
    for i in range(N):
		rik[i][ri[i]]=1
			
    	#update mus 
	for i in range(K):
		for j in range(N):
			rmus[i]=rmus[i]+rik[j][i]*data[j]
			rk[i]=rk[i]+rik[j][i]
		rmus[i]=rmus[i]/rk[i]
	
	#update covariance
	#use outer rather than dot
	for i in range(K):
		for j in range(N):
			rcoMs[i]=rcoMs[i] + rik[j][i]*numpy.outer(data[j],data[j])
		rcoMs[i]=rcoMs[i]/rk[i]
		rcoMs[i]=rcoMs[i]-numpy.outer(rmus[i],rmus[i])
    
    
    """
    count=np.zeros(K)
    
    for i in range(N):
		count[ri[i]]=count[ri[i]]+1 #calculate the count for each class
		mus[ri[i]]=mus[ri[i]]+data[i]
	#get average
	for i in range(K):
		mus[i]=mus[i]/count[i]

    """ 	
	return (rmus,rcoMs)
	

	
