#the data generation from first assignment in MATLAB
import numpy as np



def data(N,sigma):
	w=np.ones(10)/np.sqrt(10)
	a1=np.array([ 1 , 1 , 1,  1 , 1 ,-1, -1, -1, -1, -1])
	a2=np.array([-1, -1,  0 , 1 , 1 ,-1, -1, 0 , 1 , 1])
	w1=np.multiply(w,a1)
	w2=np.multiply(w,a2)
	w2=w2/np.linalg.norm(w2)
	
	x=np.zeros((4,10))
	x[1,]=x[0,]+ sigma * w1
	x[2,]=x[0,]+ sigma * w2
	x[3,]=x[0,]+ sigma * w1+ sigma * w2
	
	
	X1=x+ sigma * np.tile(w,[4,1])/2
	X2=x-sigma* np.tile(w,[4,1])/2
	
	X1=np.tile(X1,[2*N,1])
	X2=np.tile(X2,[2*N,1])
	X=np.concatenate((X1,X2),axis=0)
	
	Y1=np.ones((4*2*N,1))
	Y2=(-1) *np.ones((4*2*N,1))
	
	Y=np.concatenate((Y1,Y2),axis=0)
	Z=np.random.permutation(8*2*N)
	Z=Z[0:N]
	t=X[Z,:]
	m=np.shape(t)[0]
	n=np.shape(t)[1]
	X=X[Z,:]+ 0.2*sigma*np.random.randn(m,n)
	Y=Y[Z]
	return (X,Y)
  




