import numpy as np
import numpy
from Mahalanobis  import Mahalanobis
import matplotlib.pyplot as plt
import scipy.stats as ss
import math


def gData(vMean, mCovariance, N):
	result=list()
	for i in range(N):
		result.append(numpy.random.multivariate_normal(vMean,mCovariance))
	return result



def Mahalanobis(Data,mus,coMs):
	N=Data.shape[0]  #number of samples
	K=mus.shape[0]   # number of classes
	results=np.zeros((N,K))   #
	for i in range(N):
		for j in range(K):
			r1=Data[i]-mus[j]
			r2=np.transpose(r1)
			r3=np.dot(r1,coMs[j])
			r4=np.dot(r3,r2)
			results[i,j]=r4
	#print(results)
	return results
		
	

def Kmeans(data,mus,coMs):
	N=data.shape[0]  #number of samples
	K=mus.shape[0]  # number of classes
	ri=np.zeros(N) #similar to GMM, but here we will use hard EM
	result=Mahalanobis(data,mus,coMs)
	for i in range(N):
		maxi=result[i][0];
		zi=0
		for j in range(K):
			if result[i][j] > maxi:
				maxi=result[i][j]
				zi=j
		ri[i]=zi
	print(ri)
	return ri
	
	
	   
def KmeansUpdate(data,ri,K): 
	N=data.shape[0]
	D=data.shape[1]
	rmus=np.zeros((K,D))
	rcoMs=np.zeros((K,D,D))
    #create EM style rik so to reuse code
	for i in range(K):
		M=list()
		for j in range(N):
			if ri[j]==i:
				M.append(data[j])
		M=np.array(M)
		l=M.shape[0]
		for k in range(l):
			rmus[i]=rmus[i]+M[k]
		if l >0:
			rmus[i]=rmus[i]/l
		for k in range(l):
			x=data[k].reshape((1,D))-rmus[i]
			rcoMs[i]=rcoMs[i]+x.transpose().dot(x)
		if l>0:
			rcoMs[i]=rcoMs[i]/l
		#rcoMs[i]=rcoMs[i]-rmus[i].transpose().dot(rmus[i])

	
	print(rmus)
	print(rcoMs)
	return (rmus,rcoMs)
    
"""
    count=np.zeros(K)
    
    for i in range(N):
		count[ri[i]]=count[ri[i]]+1 #calculate the count for each class
		mus[ri[i]]=mus[ri[i]]+data[i]
	#get average
	for i in range(K):
		mus[i]=mus[i]/count[i]
""" 

	
	
def meandiff(mus,rmus):
	K=np.shape(mus)[0]
	D=np.shape(mus)[1]
	delta=rmus-mus
	result=0
	for i in range(K):
		result=result+ np.linalg.norm(delta[i])
	result=result/K
	return result
	
def KmeansIterate(data,mus,coMs):
	ri=Kmeans(data,mus,coMs)
	K=mus.shape[0]
	(rmus,rcoMs)=KmeansUpdate(data,ri,K)
	return rmus, rcoMs,ri
	

	
def Kmeansrun(data,K):
	N=np.shape(data)[0]
	D=np.shape(data)[1]
	
	#parameter initialization
	mus=np.ones((K,D))/10
	for i in range(K):
		mus[i]=mus[i]+np.random.rand(D)
	
	#covariance matrix must not be singular
	coMs=np.zeros((K,D,D))
	for i in range(K):
		coMs[i]=np.identity(D)

	pis=np.ones(K)/K
	
	ri=Kmeans(data,mus,coMs)
	(rmus,rcoMs)=KmeansUpdate(data,ri,K)
	while meandiff(mus,rmus) > 0.001 :
		mus=rmus
		coMs=rcoMs
		ri=Kmeans(data,mus,coMs)
		(rmus,rcoMs)=Kmeansupdate(data,ri,K)
	return (ri,rmus,rcoMs)


def dataGeneration():
	mu1=[1,2]
	sigma1=numpy.array([[3,1],[1,2]])
	mu2=[-1,-2]
	sigma2=numpy.array([[2,0],[0,1]])
	mu3=[3,-3]
	sigma3=numpy.array([[1,0.3],[0.3,1]])

	data1=gData(mu1,sigma1,100)	
	data2=gData(mu2,sigma2,100)	
	data3=gData(mu3,sigma3,200)	


	data=data1+data2+data3
	data=numpy.array(data) 
	return data



def main():
	 #convert to array in numpy

	#filename='data.json'
	#with open(filename,'w') as f_obj:
	#	json.dump(data,f_obj)

	#print(data)
	data=dataGeneration()

	#parameter initilization for EM
	N=400
	D=2
	K=3  #the Number of classes 
	#randomlize mu

	mus=numpy.ones((K,D))/10
	for i in range(K):
		mus[i]=mus[i]+numpy.random.rand(D)


#covariance matrix must not be singular
	coMs=numpy.zeros((K,D,D))
	for i in range(K):
		coMs[i]=numpy.identity(D)
#print(coMs)
	pis=numpy.ones(K)/K


	#plt.subplot(121)
	plt.ion()
	xaxis=list()
	
	for i in range(100):
		(rmus,rcoMs,ri)=KmeansIterate(data,mus,coMs)
		print(i)
		
		#print(rmus)
		
		xaxis.append(i)
		#plt.subplot(121)
		#print(rcoMs)
		
		plt.scatter(data[:,0],data[:,1],s=50)
		#draw the center of each gaussian 
		for i in range(K):
			plt.scatter(rmus[i,0],rmus[i,1], c=(0,0,0.8),s=200)

	#draw contour
		x = numpy.arange(-6, 6, 0.25)
		y = numpy.arange(-6, 6, 0.25)
		X, Y = numpy.meshgrid(x, y)
		for j in range(K):
			if not np.all(rmus[j]==0):
				rv= ss.multivariate_normal(rmus[j],rcoMs[j])
			#improve here
				Z=numpy.zeros((len(x),len(y)))
				for m in range(len(x)):
					for n in range(len(y)):
						Z[m][n]=rv.pdf([X[m][n],Y[m][n]])
				plt.contour(X,Y,Z,2)

		plt.pause(0.05)
		plt.clf()
	
		mus=rmus
		coMs=rcoMs
		
		
		
main()




