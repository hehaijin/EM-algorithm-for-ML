import numpy as np
import data
import Kmeans
import GMM



(D,L)=data.data(100,4)

GMM.GMMrun(D,8)
Kmeans.Kmeansrun(D,8)



