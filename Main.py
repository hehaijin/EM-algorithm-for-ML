
import numpy
from gData import gData
import matplotlib.pyplot as plt
from GMM import GMM

    
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
data=numpy.array(data)
#print(data)

N=1200
D=2
K=3
#randomlize mu
mus=numpy.ones((K,D))
for i in range(K):
	mus[i]=mus[i]+numpy.random.rand(D)
#covariance matrix must not be singular
coMs=numpy.zeros((K,D,D))
for i in range(K):
	coMs[i]=numpy.identity(D)
print(coMs)
pis=[0.3,0.3,0.4]
plt.ion()
	
for i in range(100):
	(rmus,rcoMs,rpis)=GMM(data,mus,coMs,pis)
	print(rcoMs)
	plt.scatter(data[:,0],data[:,1],s=50)
	for i in range(K):
		plt.scatter(rmus[i,0],rmus[i,1], c=(0,0,0.8),s=200)
	plt.pause(0.005)
	plt.clf()
    
	mus=rmus
	coMs=rcoMs
	pis=rpis
			
			
				
			


	
	
	
	
		
		
	
	
	


