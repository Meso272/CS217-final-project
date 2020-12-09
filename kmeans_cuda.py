
import numpy as np 
import sys
import time
from libKMCUDA import kmeans_cuda


a=np.genfromtxt(sys.argv[1],dtype=np.float32)

clusternum=int(sys.argv[2])


start=time.time()

kmeans_cuda(a, clusternum, tolerance=1e-4, init="random",yinyang_t=0.1, metric="L2", average_distance=False, device=0, verbosity=0)

end=time.time()
print(end-start)
