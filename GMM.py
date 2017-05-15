import scipy.stats
import numpy
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn import mixture


def gData(vMean, mCovariance, N):
	result=list()
	for i in range(N):
		result.append(numpy.random.multivariate_normal(vMean,mCovariance))
	return result


def GMM(data,mus,coMs,pis):
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
	return rik
	



def GMMiterate(data, mus, coMs, pis):
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
		
	#calculate log likilihood
	loglik=0
	for i in range(N):
		for j in range(K):
			loglik=loglik+rik[i][j]* numpy.log(scipy.stats.multivariate_normal.pdf(data[i],mus[j],coMs[j]))+rik[i][j]*numpy.log(pis[j]) 
		   

	
	return (rmus,rcoMs,rpis,loglik,rik)




def GMMrun(data, K):
	N=data.shape[0]
	D=data.shape[1]
	
	#parameter initialization
	mus=numpy.ones((K,D))/10
	for i in range(K):
		mus[i]=mus[i]+numpy.random.rand(D)
	
	#covariance matrix must not be singular
	coMs=numpy.zeros((K,D,D))
	for i in range(K):
		coMs[i]=numpy.identity(D)

	pis=numpy.ones(K)/K
	
	#iterate
	loglik=1;
	dlog=1;
	
	while dlog > 0.001:
		(rmus,rcoMs,rpis,logliks,rik)=GMMiterate(data,mus, coMs, pis)
		dlog=logliks-logliks
		loglik=logliks
	
	return (rik,rmus,rcoMs)
	
	
def main():
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
	data=numpy.array(data)  #convert to array in numpy

	#filename='data.json'
	#with open(filename,'w') as f_obj:
	#	json.dump(data,f_obj)

	#print(data)


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


	plt.subplot(121)
	plt.ion()

#the logliks
	logliks=list()
	xaxis=list()
	
	for i in range(100):
		(rmus,rcoMs,rpis,loglik,rik)=GMMiterate(data,mus,coMs,pis)
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
		plt.axis([0, 100, -2200, -1000])
		plt.plot(xaxis,logliks)



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
	
main()
	
	
 
