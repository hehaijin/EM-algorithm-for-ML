
import numpy 
from gData import gData
import matplotlib.pyplot as plt
from GMM import GMM
import scipy.stats as ss
import json

    
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

#filename='data.json'
#with open(filename,'w') as f_obj:
#	json.dump(data,f_obj)

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
#print(coMs)
pis=[0.3,0.3,0.4]


plt.subplot(121)
plt.ion()

#the logliks
logliks=list()
xaxis=list()
	
for i in range(100):
	(rmus,rcoMs,rpis,loglik)=GMM(data,mus,coMs,pis)
	logliks.append(loglik)
	xaxis.append(i)
	plt.subplot(121)
	#print(rcoMs)
	print(loglik)
	plt.scatter(data[:,0],data[:,1],s=50)
	#draw the center of each gaussian 
	for i in range(K):
		plt.scatter(rmus[i,0],rmus[i,1], c=(0,0,0.8),s=200)

	#draw contour
	x = numpy.arange(-6, 6, 0.25)
	y = numpy.arange(-6, 6, 0.25)
	X, Y = numpy.meshgrid(x, y)
	for j in range(K):
    
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
	pis=rpis
	
	plt.subplot(122)
	plt.axis([0, 100, -7000, -3000])
	plt.plot(xaxis,logliks)
		
				
			


	
	
	
	
		
		
	
	
	


