
import scipy.stats
import numpy
from gData import gData
import matplotlib.pyplot as plt

def GMM(data, mus, coMs, pis):
	#get data dimention
	D=data.shape[1]
	N=data.shape[0]
	K=len(mus)
	
	#initiate data structure
	rmus=numpy.zeros((K,D))
	rcoMs=numpy.zeros((K,D,D))
	rpis=numpy.zeros(K)
	rik=numpy.zeros((N,K))
	rk=numpy.zeros(K)
	
	#updating rik matrix
	for i in range(N):
		for j in range(K):
			mu=mus[j]
			co=coMs[j]
			pi=pis[j]
			rik[i][j]= pis[j]* scipy.stats.multivariate_normal.pdf(data[i],mu,co)
		sum=0
		for j in range(K):
			sum=sum+rik[i][j]
		
		for j in range(K):
			rik[i][j]=rik[i][j]/sum
	
	
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
	
	
    #update pis

	for i in range(K):
		for j in range(N):
			rpis[i]=rpis[i]+rik[j][i]
		rpis[i]=rpis[i]/N
	
	return (rmus,rcoMs,rpis)



 
