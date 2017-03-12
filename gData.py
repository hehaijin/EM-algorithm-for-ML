import numpy

def gData(vMean, mCovariance, N):
	result=list()
	for i in range(N):
		result.append(numpy.random.multivariate_normal(vMean,mCovariance))
	return result




