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
from my_sem.multigrid import twolevels, threelevels

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
#   Rx = I[1:-1, :]   # X: Dirichlet - homogeneuous Neumann
    Ry = I[1:-1, :] # Y: Dirichlet - Dirichlet
    Rmask = (Ry.T@Ry) @ np.ones((N+1,N+1)) @ ((Rx.T@Rx).T)
    return Rx,Ry,Rmask


## Conv. test
results={}
results['cg']   = np.empty((0,4)) # N, niter, err, time
results['jac']  = np.empty((0,4)) # N, niter, err, time
results['mass'] = np.empty((0,4)) # N, niter, err, time
results['fdm']  = np.empty((0,4)) # N, niter, err, time
results['cheb1']= np.empty((0,4)) # N, niter, err, time
#results['cheb2']= np.empty((0,4)) # N, niter, err, time
results['mg2j0']  = np.empty((0,4)) # N, niter, err, time
results['mg2j1']  = np.empty((0,4)) # N, niter, err, time
results['mg2c0']  = np.empty((0,4)) # N, niter, err, time
results['mg2c1']  = np.empty((0,4)) # N, niter, err, time
results['mg3j0']  = np.empty((0,4)) # N, niter, err, time
results['mg3j1']  = np.empty((0,4)) # N, niter, err, time
results['mg3c0']  = np.empty((0,4)) # N, niter, err, time
results['mg3c1']  = np.empty((0,4)) # N, niter, err, time

for N in range(3, 23):
    n = N + 1; nn = n * n

    # Set coordinates (TODO: read mesh)
    X, Y = reference_2d(N)

    # 1D operators
    z, w, Dh = semhat(N); Bh = np.diag(w) 
 
    # set 2D geom, mask 
    G, J, B = geometric_factors_2d(X, Y, w, Dh, n)
    Rx, Ry, Rmask = set_mask2d(N)

    def Ax_2d(U): # remove this later...

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


    def relax_jacobi_setup(funAx,shape): # TODO: need a class...

        omega = 2.0/3.0 # relaxtion
        Dinv = precon_jac_setup(funAx, shape, omega)

        relax_data = {"__Dinv" : Dinv, "__omega" : omega}
        return relax_data

    def relax_jacobi(funAx, u0, f, msteps, __Dinv, __omega):
        ''' 
        Usage: 
          relax_data = relax_setup(funAx,shape)
          relax(funAx, u0, f, msteps,**relax_data)
        '''
        u = u0.copy()
        for step in range(msteps):
            u += __omega * mul(__Dinv,(f - funAx(u)))
        return u

    def relax_cheb_jacobi_setup(funAx,shape):
        def dummy(): 
            return
        omega = 2.0/3.0 # relaxtion
        Dinv = precon_jac_setup(funAx, shape, omega)
        cheb_smoother = dummy
        lmin,lmax=precon_chebyshev_setup(funAx, cheb_smoother, shape
                                        ,0, lmin=0.1, lmax=1.2)

        relax_data = {"__lmin" : lmin, "__lmax" : lmax
                     ,"__Dinv" : Dinv, "__omega" : omega}
        return relax_data

    def relax_cheb_jacobi(funAx, u0, f, msteps, __lmin,__lmax,__Dinv,__omega):
        k_iter = msteps # cheb degree

        theta = 0.5*(__lmax + __lmin)
        delta = 0.5*(__lmax - __lmin)
        sigma = theta / delta
        rho = 1.0 / sigma
    
        x = u0.copy()
        r = f - funAx(x)
        r = __omega * mul(__Dinv,r)
       
#       x = 0*r
        d = 1/theta * r
        for k in range(k_iter):
            rho_prev = rho
    
            x = x + d
            r = r - __omega * mul(__Dinv,funAx(d))
            rho = 1.0 / (2.0*sigma - rho)
            d = rho*rho_prev * d + 2.0*rho / sigma * r
    
        x = x + d
        return x

    def solve_twolevels(funAx,funRelax,funRelaxSetup,tol,maxit,crsmode=0):
        if maxit<0: # use DOF
            maxit = np.sum(Rmask, dtype=np.int)
        msmth = 2

        Nf = N
        Nc = max(np.int(np.ceil(Nf/2.0)),2)
        Nc = max(np.int(np.ceil(Nf-2)),2) # N=2 is the minimum due to the Dirichlet BC, FIXME
#        print(Nf,Nc)
        vb = 0

        Ub = mul((1.0-Rmask), fun_u_exact(X,Y)) # Dirichlet BC
        b  = -mul(Rmask, funAx(Ub))

        t0 = tic()
        U, niter, res_list, Ulist = twolevels(funAx,b,set_mask2d
                           ,funRelax,funRelaxSetup,msmth,Nf,Nc
                           ,tol=tol,maxit=maxit,x0=None,ivb=vb,idumpu=False,crsmode=crsmode)
        t_elapsed = toc(t0)

        U     = U + Ub
        U_exa = fun_u_exact(X,Y)
        err   = norm_linf(U_exa-U)

#       print('mg',N,err,tol)
        return U, niter, err, t_elapsed


    def solve_threelevels(funAx,funRelax,funRelaxSetup,tol,maxit,crsmode=0,cmode=0):
        if maxit<0: # use DOF
            maxit = np.sum(Rmask, dtype=np.int)
        cmode = np.array(cmode)
        msmth = 2

        Nf = N
        if cmode.size==3:
            Nc1 = no.int(cmode[1])
            Nc2 = no.int(cmode[2])
        elif cmode==0:
            Nc1 = max(np.int(np.ceil(Nf/2.0)),2)
            Nc2 = max(np.int(np.ceil(Nc1/2.0)),2)
        elif cmode==1:
            Nc1 = max(np.int(np.ceil(Nf-2)),2)
            Nc2 = max(np.int(np.ceil(Nc1-2)),2)
        print(Nf,Nc1,Nc2)
        vb = 0

        Ub = mul((1.0-Rmask), fun_u_exact(X,Y)) # Dirichlet BC
        b  = -mul(Rmask, funAx(Ub))

        t0 = tic()
        U, niter, res_list, Ulist = threelevels(funAx,b,set_mask2d
                           ,funRelax,funRelaxSetup,msmth,Nf,Nc1,Nc2
                           ,tol=tol,maxit=maxit,x0=None,ivb=vb,idumpu=False,crsmode=crsmode)
        t_elapsed = toc(t0)

        U     = U + Ub
        U_exa = fun_u_exact(X,Y)
        err   = norm_linf(U_exa-U)

#       print('mg',N,err,tol)
        return U, niter, err, t_elapsed


    # Setup preconditioner (store into module-wise global memory)
    Minv = 1.0 / mul(B, J)
    precon_mass_setup(Minv)

    omega = 2.0/3.0 # relaxtion
    Dinv = precon_jac_setup(Ax_2d, X.shape, omega)

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

    tag = 'mg2j0' # 2-lv jac, crsmode=0
    U, niter, err, t_elapsed = solve_twolevels(Ax_2d,relax_jacobi,relax_jacobi_setup
                                              ,tol,maxit,crsmode=0)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

    tag = 'mg2j1' # 2-lv jac, crsmode=1
    U, niter, err, t_elapsed = solve_twolevels(Ax_2d,relax_jacobi,relax_jacobi_setup
                                              ,tol,maxit,crsmode=1)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

    tag = 'mg2c0' # 2-lv jac+cheb, crsmode=0
    U, niter, err, t_elapsed = solve_twolevels(Ax_2d,relax_cheb_jacobi,relax_cheb_jacobi_setup
                                              ,tol,maxit,crsmode=0)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

    tag = 'mg2c1' # 2-lv jac+cheb, crsmode=1
    U, niter, err, t_elapsed = solve_twolevels(Ax_2d,relax_cheb_jacobi,relax_cheb_jacobi_setup
                                              ,tol,maxit,crsmode=1)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

    tag = 'mg3j0' # 2-lv jac, crsmode=0
    U, niter, err, t_elapsed = solve_threelevels(Ax_2d,relax_jacobi,relax_jacobi_setup
                                                ,tol,maxit,crsmode=0)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

    tag = 'mg3j1' # 3-lv jac, crsmode=1
    U, niter, err, t_elapsed = solve_threelevels(Ax_2d,relax_jacobi,relax_jacobi_setup
                                                ,tol,maxit,crsmode=1)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

    tag = 'mg3j0' # 2-lv jac, crsmode=0
    U, niter, err, t_elapsed = solve_threelevels(Ax_2d,relax_jacobi,relax_jacobi_setup
                                                ,tol,maxit,crsmode=0)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

    tag = 'mg3j1' # 3-lv jac, crsmode=1
    U, niter, err, t_elapsed = solve_threelevels(Ax_2d,relax_jacobi,relax_jacobi_setup
                                                ,tol,maxit,crsmode=1)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed]], axis=0)

## plots and saves
def plot_aux(results,idx,stry,ifsave,strf):
    ax = plt.figure().gca()
    plt.semilogy(results['cg']   [:,0], results['cg']   [:,idx],"-o", label="cg")
    plt.semilogy(results['mass'] [:,0], results['mass'] [:,idx],"-o", label="pcg(mass)")
    plt.semilogy(results['jac']  [:,0], results['jac']  [:,idx],"-o", label="pcg(jacobi)")
    plt.semilogy(results['fdm']  [:,0], results['fdm']  [:,idx],"-o", label="pcg(fdm)")
    plt.semilogy(results['cheb1'][:,0], results['cheb1'][:,idx],"-o", label="pcg(cheb-jac)")
#   plt.semilogy(results['cheb2'][:,0], results['cheb2'][:,idx],"-o", label="pcg(cheb-mass)")
    plt.semilogy(results['mg2j0'][:,0], results['mg2j0'][:,idx],"-o", label="2-lv(jac0)")
    plt.semilogy(results['mg2j1'][:,0], results['mg2j1'][:,idx],"-o", label="2-lv(jac1)")
    plt.semilogy(results['mg2c0'][:,0], results['mg2c0'][:,idx],"-o", label="2-lv(cheb-jac0)")
    plt.semilogy(results['mg2c1'][:,0], results['mg2c1'][:,idx],"-o", label="2-lv(cheb-jac1)")
    plt.semilogy(results['mg3j0'][:,0], results['mg3j0'][:,idx],"-o", label="3-lv(jac0)")
    plt.semilogy(results['mg3j1'][:,0], results['mg3j1'][:,idx],"-o", label="3-lv(jac1)")
    plt.semilogy(results['mg3c0'][:,0], results['mg3c0'][:,idx],"-o", label="3-lv(cheb-jac0)")
    plt.semilogy(results['mg3c1'][:,0], results['mg3c1'][:,idx],"-o", label="3-lv(cheb-jac1)")
    plt.title("tol="+str(tol), fontsize=20); plt.legend(loc=0)
    plt.xlim(1, N + 1); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("N - order", fontsize=16); plt.ylabel(stry, fontsize=16)
    if(ifsave):
        plt.savefig(strf, bbox_inches="tight")

plot_aux(results,3,"Elapsed Time (s)",ifsave,"elapsed_pcg.pdf")
plot_aux(results,1,"# iterations",ifsave,"niter_pcg.pdf")
plot_aux(results,2,"max. abs. error",ifsave,"err_pcg.pdf")


fig = plt.figure(); ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, U)
plt.title("Solution profile", fontsize=20)
plt.xlabel(r'X', fontsize=16); plt.ylabel(r'Y', fontsize=16)
if(ifsave):
    plt.savefig("solution_surf.pdf", bbox_inches="tight")



if plot_on:
    plt.show()
