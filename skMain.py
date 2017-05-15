from sklearn import mixture
import numpy as numpy
from gData import gData


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

