import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from my_sem.linear_solvers import arnoldi_iteration

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

__lmin = None
__lmax = None
__k_cheb = None
__Ax_cheb = None
__Sx_cheb = None

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


def precon_chebyshev_setup(funAx, fun_smoother, shape, k, lmin=0.1, lmax=1.2):
    global __lmin, __lmax, __k_cheb, __Ax_cheb, __Sx_cheb

    # Estimate the maximum eigenvalue of A via arnoldi(10)
    np.random.seed(11) # for reproducible
    b = np.random.random_sample(shape)
    Q,H = arnoldi_iteration(funAx, b, 10)

    l_est,_ = spla.eigs(H[:-1,:], k=1, which='LM') # 10 by 10 matrix
    l_est = l_est.real

    # debug: build full matrix and find max eigenvalues (expansive)
#    n, m = X.shape
#    AA = np.zeros((m*n,m*n))
#    for j in range(m*n):
#      for i in range(n*n):
#        tmpi = np.zeros((m*n))
#        tmpj = np.zeros((m*n))
#        tmpi[i]=1
#        tmpj[j]=1
#        AA[j,i] = np.sum(tmpi*Ax_2d(tmpj.reshape((n,m))).reshape((m*n,)))
#
#    l_est2,_ = spla.eigs(AA, k=1, which='LM')
#    l_est2 =  l_est2.real
#    print(X.shape,l_est,l_est2)

    __lmin = lmin*l_est
    __lmax = lmax*l_est
    __k_cheb = k
    __Ax_cheb = funAx
    __Sx_cheb = fun_smoother
    return lmin,lmax


def precon_chebyshev(r):
    
    theta = 0.5*(__lmax + __lmin)
    delta = 0.5*(__lmax - __lmin)
    sigma = theta / delta
    rho = 1.0 / sigma

    r = __Sx_cheb(r)
    x = 0*r
    d = 1/theta * r
    for k in range(__k_cheb):
        rho_prev = rho  

        x = x + d
        r = r - __Sx_cheb(__Ax_cheb(d))
        rho = 1.0 / (2.0*sigma - rho)
        d = rho*rho_prev * d + 2.0*rho / sigma * r

    r = x + d 
    return r
