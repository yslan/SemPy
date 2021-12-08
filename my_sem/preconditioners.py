import numpy as np
import scipy.linalg as sla

# Non-multigrid options


# Global varibles survives in this modules
# TODO: we should use some data structure, so
#     call solver.setup()
#     run solver.solve()

__Minv = None

__omega = None
__Dinv = None

__dinv = None
__SRx = None
__SRy = None
__Rmask = None


# TODO: put assert functions here so pcg can check (once) before calling it


def precon_mass_setup(Minv):
    global __Minv 
    __Minv = Minv

def precon_mass(r):
    return __Minv * r


def precon_jac_setup(funAx, x, omega): # fetch diaginal of matrix A
# TODO: build diagonal directly without loops
    global __omega, __Dinv
    n, m = x.shape
    DD = np.zeros((m,n))
    for j in range(m):
      for i in range(n):
        tmp = np.zeros((m,n))
        tmp[j,i]=1
        DD[j,i] = np.sum(tmp*funAx(tmp))
    __Dinv = DD
    __Dinv[__Dinv!=0] = 1.0 / __Dinv[__Dinv!=0]
    __omega = omega
    return __Dinv

def precon_jac(r):
    return __omega * __Dinv * r


def precon_fdm_2d_setup(Bh, Dh, Rx, Ry, Rmask): # FIXME: use helmholtz form: a*A + b*B
    global __Rmask, __SRx, __SRy, __dinv

    Ah = Dh.T @ Bh @ Dh
    Ah = 0.5 * (Ah + Ah.T) # To make sure A is numerically s.p.d.

    Ax = Rx @ Ah @ Rx.T
    Ay = Ry @ Ah @ Ry.T
    Bx = Rx @ Bh @ Rx.T
    By = Ry @ Bh @ Ry.T

    Lx, Sx = sla.eigh(Ax, Bx)
    Ly, Sy = sla.eigh(Ay, By)

    __SRx = Rx.T @ Sx
    __SRy = Ry.T @ Sy
    __SRx = __SRx.real
    __SRy = __SRy.real

    Lx = Lx.real
    Ly = Ly.real
    Lx = Lx.reshape((Lx.shape[0],1))
    Ly = Ly.reshape((Ly.shape[0],1))
    ex = 1+0*Lx
    ey = 1+0*Ly
    D =  Ly @ ex.T + ey @ Lx.T 
    __dinv = 1.0 / D

    __Rmask = Rmask
    return __SRx, __SRy, __dinv


def precon_fdm_2d(U):
    U = __Rmask*U
    U = __SRy @ ( __dinv*(__SRy.T @ U @ __SRx) ) @ __SRx.T
    return __Rmask*U
