import numpy as np
from ica import ica1
import time

NSUB = 1000
NCOMP = 100
NVOX = 50000

true_a = np.random.normal(0,1,NSUB*NCOMP).reshape((NSUB,NCOMP))
true_s = np.random.logistic(0,1,NCOMP*NVOX).reshape((NCOMP, NVOX))

true_x = np.dot(true_a, true_s) + np.random.normal(0,1, NSUB*NVOX).reshape((NSUB,NVOX))
import time

start = time.time()
A,S  = ica1(true_x,NCOMP)
end = time.time()
print end - start
