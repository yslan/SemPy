import numpy as np
import time

def norm_linf(u):
    return np.max(np.abs(u))
def norm_L2(u,wt):
    return np.sqrt(np.sum(u*u*wt))


__tprev = None

def tic():
    global __tprev
    __tprev = time.process_time()
    return __tprev

def toc(t0=None):
    t1 = time.process_time() # stop timer first
    if t0 is None:
      t0 = __tprev
    return t1 - t0
