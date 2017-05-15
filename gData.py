import numpy

def gData(vMean, mCovariance, N):
	result=list()
	for i in range(N):
		result.append(numpy.random.multivariate_normal(vMean,mCovariance))
	return result





def gData(vMean, mCovariance, N):
	result=list()
	for i in range(N):
		result.append(numpy.random.multivariate_normal(vMean,mCovariance))
	return result




def updateMuVa(data,rik):
	N=rik.shape[0]
	K=rik.shape[1]
	
		
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
	return (rmus,rcoMs)
