import numpy as np 
import sys
from io import StringIO
from sklearn import cluster
import time


a=np.genfromtxt(sys.argv[1],dtype=np.float32)
clusternum=int(sys.argv[2])

alg=sys.argv[3]
start=time.time()

labels=cluster.KMeans(n_clusters=clusternum,max_iter=300,init='random',n_init=1,tol=0,algorithm=alg).fit_predict(a)
end=time.time()
print(end-start)

