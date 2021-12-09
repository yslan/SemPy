import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from numpy import multiply as mul

from my_sem.gen_semhat import semhat
from my_sem.gen_geom import geometric_factors_2d
from my_sem.linear_solvers import cg, pcg, arnoldi_iteration
from my_sem.preconditioners import (
    precon_mass_setup,
    precon_mass,
    precon_jac_setup,
    precon_jac,
    precon_fdm_2d_setup,
    precon_fdm_2d,
    precon_chebyshev_setup,
    precon_chebyshev
)
from my_sem.util import tic, toc, norm_linf

from sempy.meshes.box import reference_2d

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# 2D Poisson solver:  -nabla^2 u = 0
# Available BCs:
#    inhomogenuous Dirichlet 
#    homogenuous Nuemann
# Notes:
#    single element
#    non-deformed geometry (domain: [-1,1]^2)
#    deformed geometry (not tested)


# Data structures for d-dim arrays:
#     X: d-dim array, X[i,j] or X[i,j,k], C like array, d/dx = X @ Dh.T
#    xx: 1d prolonged array, index xx[k*n*n + j*n + i]
#     x: if it doesn't matter
# Transformation:
#     X = xx.reshape((n,n,n))
#    xx = reshape(X,(n*n,))
# Common variables:
#     N: polynomial order
#     n: N+1, # grid points in 1D(local)
#    nn: n*n, # grid points in 2D or 3D


## User inputs:
plot_on = 1
ifsave = 0

# Exact solution
def fun_u_exact(x,y): 
    a=np.pi/2;
    u = mul( np.exp(a * y), np.sin(a * x)) # du/dx(x=1) = 0
    return np.array(u)

# Boundary condition 
def set_mask2d(N): # TODO: add input to control dffernet BC
    I = np.identity(N+1, dtype=np.float64)
    Rx = I[1:, :]   # X: Dirichlet - homogeneuous Neumann
    Ry = I[1:-1, :] # Y: Dirichlet - Dirichlet
    Rmask = (Ry.T@Ry) @ np.ones((n,n)) @ ((Rx.T@Rx).T)
    return Rx,Ry,Rmask


## Conv. test
results={}
results['cg']   = np.empty((0,4)) # N, niter, err, time
results['jac']  = np.empty((0,4)) # N, niter, err, time
results['mass'] = np.empty((0,4)) # N, niter, err, time
results['fdm']  = np.empty((0,4)) # N, niter, err, time
results['cheb1']= np.empty((0,4)) # N, niter, err, time
#results['cheb2']= np.empty((0,4)) # N, niter, err, time

for N in range(3, 23):
    n = N + 1; nn = n * n

    # Set coordinates (TODO: read mesh)
    X, Y = reference_2d(N)

    # 1D operators
    z, w, Dh = semhat(N); Bh = np.diag(w) 
 
    # set 2D geom, mask 
    G, J, B = geometric_factors_2d(X, Y, w, Dh, n)
    Rx, Ry, Rmask = set_mask2d(N)

    def Ax_2d(U): 

        Ux = U @ Dh.T
        Uy = Dh @ U
    
        Wx = mul(G[0, 0, :, :],Ux) + mul(G[0, 1, :, :],Uy)
        Wy = mul(G[1, 0, :, :],Ux) + mul(G[1, 1, :, :],Uy)
   
        W = Wx @ Dh + Dh.T @ Wy

        return mul(Rmask,W)

    def solve_cg(funAx,tol,maxit,vb=0): # also reads: X,Y,Rmask
      if maxit<0: # use DOF
        maxit = np.sum(Rmask, dtype=np.int)

      Ub = mul((1.0-Rmask), fun_u_exact(X,Y)) # Dirichlet BC
      b  = - mul(Rmask, funAx(Ub))

      t0 = tic()
      U, niter = cg(funAx, b, wt=Rmask, tol=tol, maxit=maxit, verbose=vb)
      t_elapsed = toc(t0)

      U     = U + Ub
      U_exa = fun_u_exact(X,Y)
      err   = norm_linf(U_exa-U)

      return U, niter, err, t_elapsed

    def solve_pcg(funAx,funPrecon,tol,maxit,vb=0): # also reads: X,Y,Rmask
      if maxit<0: # use DOF
        maxit = np.sum(Rmask, dtype=np.int) 
      if vb==1:
        print('maxit',maxit,tol)

      Ub = mul((1.0-Rmask), fun_u_exact(X,Y)) # Dirichlet BC
      b  = -mul(Rmask, funAx(Ub))

      t0 = tic()
      U, niter = pcg(funAx, funPrecon, b, wt=Rmask, tol=tol, maxit=maxit, verbose=vb)
      t_elapsed = toc(t0)

      U     = U + Ub
      U_exa = fun_u_exact(X,Y)
      err   = norm_linf(U_exa-U)

      return U, niter, err, t_elapsed


    # Setup preconditioner (store into module-wise global memory)
    Minv = 1.0 / mul(B, J)
    precon_mass_setup(Minv)

    omega = 2.0/3.0 # relaxtion
    Dinv = precon_jac_setup(Ax_2d, X, omega)

    precon_fdm_2d_setup(Bh, Dh, Rx, Ry, Rmask) 

    # TODO: fit chebyshev parameters
    cheb_smoother = precon_jac # jacobi-chebyshev
    k_iter = 3
    precon_chebyshev_setup(Ax_2d, cheb_smoother, X.shape, k_iter, lmin=0.1, lmax=1.2)

    # Main solves
    tol = 1e-8
    maxit = -1 # maxit = DOF

    tag = 'cg'
    U, niter, err, t_elapsed = solve_cg(Ax_2d,tol,maxit)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

    tag = 'mass'
    U, niter, err, t_elapsed = solve_pcg(Ax_2d,precon_mass,tol,maxit)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

    tag = 'jac'
    U, niter, err, t_elapsed = solve_pcg(Ax_2d,precon_jac,tol,maxit)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

    tag = 'fdm'
    U, niter, err, t_elapsed = solve_pcg(Ax_2d,precon_fdm_2d,tol,maxit)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

    tag = 'cheb1' # cheb + jac
    U, niter, err, t_elapsed = solve_pcg(Ax_2d,precon_chebyshev,tol,maxit)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

#    tag = 'cheb2' # cheb + mass lmax unbdd
#    cheb_smoother = precon_mass 
#    k_iter = 0
#    precon_chebyshev_setup(Ax_2d, cheb_smoother, X.shape, k_iter, lmin=0.1, lmax=1.2)
#    U, niter, err, t_elapsed = solve_pcg(Ax_2d,precon_chebyshev,tol,maxit,vb=1)
#    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)
#    print(niter)

## plots and saves
ax = plt.figure().gca()
plt.semilogy(results['cg']   [:,0], results['cg']   [:,3], "-o", label="cg")
plt.semilogy(results['mass'] [:,0], results['mass'] [:,3], "-o", label="pcg(mass)")
plt.semilogy(results['jac']  [:,0], results['jac']  [:,3], "-o", label="pcg(jacobi)")
plt.semilogy(results['fdm']  [:,0], results['fdm']  [:,3], "-o", label="pcg(fdm)")
plt.semilogy(results['cheb1'][:,0], results['cheb1'][:,3], "-o", label="pcg(cheb-jac)")
#plt.semilogy(results['cheb2'][:,0], results['cheb2'][:,3], "-o", label="pcg(cheb-mass)")
plt.title("tol="+str(tol), fontsize=20); plt.legend(loc=0)
plt.xlim(1, N + 1); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("N - order", fontsize=16); plt.ylabel("Elapsed Time (s)", fontsize=16)
if(ifsave):
    plt.savefig("elapsed_pcg.pdf", bbox_inches="tight")

ax = plt.figure().gca()
plt.semilogy(results['cg']   [:,0], results['cg']   [:,1], "-o", label="cg")
plt.semilogy(results['mass'] [:,0], results['mass'] [:,1], "-o", label="pcg(mass)")
plt.semilogy(results['jac']  [:,0], results['jac']  [:,1], "-o", label="pcg(jacobi)")
plt.semilogy(results['fdm']  [:,0], results['fdm']  [:,1], "-o", label="pcg(fdm)")
plt.semilogy(results['cheb1'][:,0], results['cheb1'][:,1], "-o", label="pcg(cheb-jac)")
#plt.semilogy(results['cheb2'][:,0], results['cheb2'][:,1], "-o", label="pcg(cheb-mass)")
plt.title("tol="+str(tol), fontsize=20); plt.legend(loc=0)
plt.xlim(2, N + 1); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("N - order", fontsize=16); plt.ylabel("# iterations", fontsize=16)
if(ifsave):
    plt.savefig("niter_pcg.pdf", bbox_inches="tight")

ax = plt.figure().gca()
plt.semilogy(results['cg']   [:,0], results['cg']   [:,2], "-o", label="cg")
plt.semilogy(results['mass'] [:,0], results['mass'] [:,2], "-o", label="pcg(mass)")
plt.semilogy(results['jac']  [:,0], results['jac']  [:,2], "-o", label="pcg(jacobi)")
plt.semilogy(results['fdm']  [:,0], results['fdm']  [:,2], "-o", label="pcg(fdm)")
plt.semilogy(results['cheb1'][:,0], results['cheb1'][:,2], "-o", label="pcg(cheb-jac)")
#plt.semilogy(results['cheb2'][:,0], results['cheb2'][:,2], "-o", label="pcg(cheb-mass)")
plt.title("tol="+str(tol), fontsize=20); plt.legend(loc=0)
plt.xlim(2, N + 1); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("N - order", fontsize=16); plt.ylabel("max. abs. error", fontsize=16)
if(ifsave):
    plt.savefig("err_pcg.pdf", bbox_inches="tight")

fig = plt.figure(); ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, U)
plt.title("Solution profile", fontsize=20)
plt.xlabel(r'X', fontsize=16); plt.ylabel(r'Y', fontsize=16)
if(ifsave):
    plt.savefig("solution_surf.pdf", bbox_inches="tight")


if plot_on:
    plt.show()
