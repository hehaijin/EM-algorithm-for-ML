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
			#print(r4)
			#print('\n')
			results[i]=r4

	return results
		
	
	
#print(Mahalanobis(np.array([1,2]),np.array([1,1]),np.array([[1,0,],[0,3]])))



def skmain():
	mu1=[1,2]
	sigma1=numpy.array([[3,1],[1,2]])
	mu2=[-1,-2]
	sigma2=numpy.array([[2,0],[0,1]])
	mu3=[3,-3]
	sigma3=numpy.array([[1,0.3],[0.3,1]])

	data1=gData(mu1,sigma1,300)	
	data2=gData(mu2,sigma2,300)	
	data3=gData(mu3,sigma3,600)	


	data=data1+data2+data3
	data=numpy.array(data)  #convert to array in numpy

	n_samples=3
	cv_type='full'

	gmm=mixture.GaussianMixture(n_samples,cv_type)

	gmm.fit(data)
	print(gmm.predict_proba(data).size)
