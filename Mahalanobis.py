
import numpy as np

def Mahalanobis(D,mus,coMs):
	N=D.shape[0]  #number of samples
	K=mus.shape[0]   # number of classes
	results=np.zeros((N,K))   #
	for i in range(N):
		for j in range(K):
			r1=D[i]-mus[j]
			r2=np.transpose(r1)
			r3=np.dot(r1,coMs[j])
			r4=np.dot(r3,r2)
			print(r4)
			print('\n')
			results[i]=r4

	return results
		
	
	
#print(Mahalanobis(np.array([1,2]),np.array([1,1]),np.array([[1,0,],[0,3]])))
