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

'''
   This is to runs and plot lots of 2 levels cases
'''


## User inputs:
use_jac_as_relax = 0 # 0=jac, 1=cheb+jac
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
#   Rx = I[1:, :]   # X: Dirichlet - homogeneuous Neumann
    Rx = I[1:-1, :] # X: Dirichlet - Dirichlet
    Ry = I[1:-1, :] # Y: Dirichlet - Dirichlet
    Rmask = (Ry.T@Ry) @ np.ones((N+1,N+1)) @ ((Rx.T@Rx).T)
    return Rx,Ry,Rmask


## Conv. test
results={}
results['cg']     = np.empty((0,5)) # N, niter, err, t_solve, t_elapsed
results['jac']    = np.empty((0,5)) # N, niter, err, t_solve, t_elapsed
results['cheb']   = np.empty((0,5)) # N, niter, err, t_solve, t_elapsed

results['mg3Ihs']  = np.empty((0,5)) # N, niter, err, t_solve, t_elapsed
results['mg3Ihb']  = np.empty((0,5)) # N, niter, err, t_solve, t_elapsed
results['mg3Ims']  = np.empty((0,5)) # N, niter, err, t_solve, t_elapsed
results['mg3Imb']  = np.empty((0,5)) # N, niter, err, t_solve, t_elapsed
results['mg3Bhs']  = np.empty((0,5)) # N, niter, err, t_solve, t_elapsed
results['mg3Bhb']  = np.empty((0,5)) # N, niter, err, t_solve, t_elapsed
results['mg3Bms']  = np.empty((0,5)) # N, niter, err, t_solve, t_elapsed
results['mg3Bmb']  = np.empty((0,5)) # N, niter, err, t_solve, t_elapsed

for N in range(3, 41):
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

    def solve_twolevels(funAx,funRelax,funRelaxSetup,tol,maxit,crsmode=1,cmode=0):
        '''
           crsmode=1:  J_f2c = Jc2f.T, Ac = interp (Af)
           crsmode=2:  J_f2c = Jc2f.T, Ac = build_on(Xc,Yc)
           crsmode=-1: J_f2c = interp_setup(Nc,Nf), Ac = interp (Af)
           crsmode=-2: J_f2c = interp_setup(Nc,Nf), Ac = build_on(Xc,Yc)

           cmode=0: N -> N/2
           cmode=1: N -> N-2
           cmode=[Nf,Nc]: Nf->Nc
        '''
        if maxit<0: # use DOF
            maxit = np.sum(Rmask, dtype=np.int)
        cmode = np.array(cmode)
        msmth = 2 # iter(jac) or deg(cheb-jac 1)
        vb = 0

        Nf = N
        if cmode.size==2:
            Nc = no.int(cmode[1])
        elif cmode==0:
            Nc = max(np.int(np.ceil(Nf/2.0)),2)
        elif cmode==1:
            Nc = max(np.int(np.ceil(Nf-2)),2)
#       Nc=2 # force coarse level
        print(Nf,Nc)

        Ub = mul((1.0-Rmask), fun_u_exact(X,Y)) # Dirichlet BC
        b  = -mul(Rmask, funAx(Ub))

        t0 = tic()
        U, niter, res_list, Ulist, t_solve = twolevels(funAx,b,set_mask2d
                           ,funRelax,funRelaxSetup,msmth,Nf,Nc
                           ,tol=tol,maxit=maxit,x0=None,ivb=vb,idumpu=False,crsmode=crsmode)
        t_elapsed = toc(t0)

        U     = U + Ub
        U_exa = fun_u_exact(X,Y)
        err   = norm_linf(U_exa-U)

        return U, niter, err, t_solve, t_elapsed


    def solve_threelevels(funAx,funRelax,funRelaxSetup,tol,maxit,crsmode=1,cmode=0):
        '''
           crsmode=1:  J_f2c = Jc2f.T, Ac = interp (Af)
           crsmode=2:  J_f2c = Jc2f.T, Ac = build_on(Xc,Yc)
           crsmode=-1: J_f2c = interp_setup(Nc,Nf), Ac = interp (Af)
           crsmode=-2: J_f2c = interp_setup(Nc,Nf), Ac = build_on(Xc,Yc)

           cmode=0: N -> N/2
           cmode=1: N -> N-2
           cmode=[Nf,Nc]: Nf->Nc
        '''
        if maxit<0: # use DOF
            maxit = np.sum(Rmask, dtype=np.int)
        cmode = np.array(cmode)
        msmth = 2 # iter(jac) or deg(cheb-jac 1)
        vb = 0

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
#       Nc2=2 # force coarse level
        print(Nf,Nc1,Nc2)


        Ub = mul((1.0-Rmask), fun_u_exact(X,Y)) # Dirichlet BC
        b  = -mul(Rmask, funAx(Ub))

        t0 = tic()
        U, niter, res_list, Ulist, t_solve = threelevels(funAx,b,set_mask2d
                           ,funRelax,funRelaxSetup,msmth,Nf,Nc1,Nc2
                           ,tol=tol,maxit=maxit,x0=None,ivb=vb,idumpu=False,crsmode=crsmode)
        t_elapsed = toc(t0)

        U     = U + Ub
        U_exa = fun_u_exact(X,Y)
        err   = norm_linf(U_exa-U)

#       print('mg',N,err,tol)
        return U, niter, err, t_solve, t_elapsed


    # Setup preconditioner (store into module-wise global memory)
    Minv = 1.0 / mul(B, J)
    precon_mass_setup(Minv)

    omega = 2.0/3.0 # relaxtion
    Dinv = precon_jac_setup(Ax_2d, X.shape, omega)

    precon_fdm_2d_setup(Bh, Dh, Rx, Ry, Rmask) 

    cheb_smoother = precon_jac # jacobi-chebyshev
    k_iter = 3
    precon_chebyshev_setup(Ax_2d, cheb_smoother, X.shape, k_iter, lmin=0.1, lmax=1.2)

    # Main solves
    tol = 1e-8
    maxit = -1 # maxit = DOF

    tag = 'cg'
    U, niter, err, t_elapsed = solve_cg(Ax_2d,tol,maxit)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed, t_elapsed]], axis=0)

    tag = 'jac'
    U, niter, err, t_elapsed = solve_pcg(Ax_2d,precon_jac,tol,maxit)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed, t_elapsed]], axis=0)

    tag = 'cheb' # cheb + jac
    U, niter, err, t_elapsed = solve_pcg(Ax_2d,precon_chebyshev,tol,maxit)
    results[tag] = np.append(results[tag], [[N, niter, err, t_elapsed, t_elapsed]], axis=0)


    ## switch relaxation here:
    if use_jac_as_relax == 1:
      str_lg = 'jac'
      relax = relax_jacobi
      relaxSetup = relax_jacobi_setup
    else:
      str_lg = 'cheb+jac'
      relax = relax_cheb_jacobi
      relaxSetup = relax_cheb_jacobi_setup

    tag = 'mg3Ihs'
    U, niter, err, t_solve, t_elapsed = solve_threelevels(Ax_2d,relax,relaxSetup
                                              ,tol,maxit,crsmode=1, cmode=0)
    results[tag] = np.append(results[tag], [[N, niter, err, t_solve, t_elapsed]], axis=0)

    tag = 'mg3Ihb' 
    U, niter, err, t_solve, t_elapsed = solve_threelevels(Ax_2d,relax,relaxSetup
                                              ,tol,maxit,crsmode=-1,cmode=0)
    results[tag] = np.append(results[tag], [[N, niter, err, t_solve, t_elapsed]], axis=0)

    tag = 'mg3Ims'
    U, niter, err, t_solve, t_elapsed = solve_threelevels(Ax_2d,relax,relaxSetup
                                              ,tol,maxit,crsmode=1, cmode=1)
    results[tag] = np.append(results[tag], [[N, niter, err, t_solve, t_elapsed]], axis=0)

    tag = 'mg3Imb'
    U, niter, err, t_solve, t_elapsed = solve_threelevels(Ax_2d,relax,relaxSetup
                                              ,tol,maxit,crsmode=-1,cmode=1)
    results[tag] = np.append(results[tag], [[N, niter, err, t_solve, t_elapsed]], axis=0)

    tag = 'mg3Bhs'
    U, niter, err, t_solve, t_elapsed = solve_threelevels(Ax_2d,relax,relaxSetup
                                              ,tol,maxit,crsmode=2, cmode=0)
    results[tag] = np.append(results[tag], [[N, niter, err, t_solve, t_elapsed]], axis=0)

    tag = 'mg3Bhb'
    U, niter, err, t_solve, t_elapsed = solve_threelevels(Ax_2d,relax,relaxSetup
                                              ,tol,maxit,crsmode=-2,cmode=0)
    results[tag] = np.append(results[tag], [[N, niter, err, t_solve, t_elapsed]], axis=0)

    tag = 'mg3Bms'
    U, niter, err, t_solve, t_elapsed = solve_threelevels(Ax_2d,relax,relaxSetup
                                              ,tol,maxit,crsmode=2, cmode=1)
    results[tag] = np.append(results[tag], [[N, niter, err, t_solve, t_elapsed]], axis=0)

    tag = 'mg3Bmb'
    U, niter, err, t_solve, t_elapsed = solve_threelevels(Ax_2d,relax,relaxSetup
                                              ,tol,maxit,crsmode=-2, cmode=1)
    results[tag] = np.append(results[tag], [[N, niter, err, t_solve, t_elapsed]], axis=0)


## plots and saves

# gen some colors...
import colorsys

def get_N_HexCol(N):
  HSV_tuples = [(x*1.0/N, 0.8, 0.8) for x in range(N)]
  hex_out = []
  for rgb in HSV_tuples:
      rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
      hex_out.append('#%02x%02x%02x' % tuple(rgb))
  return hex_out

def get_my_colors(Nplt,cc_id=None):
  #cc_id = [0,1,2,3,4,5]
  cc = get_N_HexCol(Nplt)
  if cc_id is not None:
    c_candy = get_N_HexCol(Nplt)
    for ii in range(Nplt):
      cc[ii] = c_candy[cc_id[ii]]
  return cc

ccmap=get_my_colors(20)


def plot_aux(data,idx,cc,stry,ifsave,strf):
    ax = plt.figure().gca()
    pfun=plt.semilogy
    pfun(data['cg']    [:,0],data['cg']    [:,idx],"-s", color=cc[0], label="cg")
    pfun(data['jac']   [:,0],data['jac']   [:,idx],"-s", color=cc[1], label="pcg(jacobi)")
    pfun(data['cheb']  [:,0],data['cheb']  [:,idx],"-s", color=cc[3], label="pcg(cheb-jac)")
    pfun(data['mg3Ihs'][:,0],data['mg3Ihs'][:,idx],":o", color=cc[4], label="mg3Ihs (3lv "+str_lg+")") 
    pfun(data['mg3Ihb'][:,0],data['mg3Ihb'][:,idx],":^", color=cc[5], label="mg3Ihb (3lv "+str_lg+")")
    pfun(data['mg3Ims'][:,0],data['mg3Ims'][:,idx],"-o", color=cc[6], label="mg3Ims (3lv "+str_lg+")")
    pfun(data['mg3Imb'][:,0],data['mg3Imb'][:,idx],"-^", color=cc[7], label="mg3Imb (3lv "+str_lg+")")
    pfun(data['mg3Bhs'][:,0],data['mg3Bhs'][:,idx],":o", color=cc[8], label="mg3Bhs (3lv "+str_lg+")")
    pfun(data['mg3Bhb'][:,0],data['mg3Bhb'][:,idx],":^", color=cc[9], label="mg3Bhb (3lv "+str_lg+")")
    pfun(data['mg3Bms'][:,0],data['mg3Bms'][:,idx],"-o", color=cc[10],label="mg3Bms (3lv "+str_lg+")")
    pfun(data['mg3Bmb'][:,0],data['mg3Bmb'][:,idx],"-^", color=cc[11],label="mg3Bmb (3lv "+str_lg+")")
    plt.title("tol="+str(tol), fontsize=20); plt.legend(loc=0)
    plt.xlim(1, N + 1); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("N - order", fontsize=16); plt.ylabel(stry, fontsize=16)
    if(ifsave):
        plt.savefig(strf, bbox_inches="tight")

plot_aux(results,3,ccmap,"Elapsed Time (s)",ifsave,"elapsed_pcg.pdf")
plot_aux(results,1,ccmap,"# iterations",ifsave,"niter_pcg.pdf")
plot_aux(results,2,ccmap,"max. abs. error",ifsave,"err_pcg.pdf")


if plot_on:
    plt.show()
