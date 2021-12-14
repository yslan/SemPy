import numpy as np
import scipy.linalg as sla
from numpy import multiply as mul

from sempy.meshes.box import reference_2d
from my_sem.gen_semhat import semhat, gauss_lobatto
from my_sem.interp1d import interp_mat
from my_sem.gen_geom import geometric_factors_2d
from my_sem.util import tic,toc

def fun_MG_Ax(U,Dh,G,Rmask):

    Ux = U @ Dh.T
    Uy = Dh @ U

    Wx = mul(G[0, 0, :, :],Ux) + mul(G[0, 1, :, :],Uy)
    Wy = mul(G[1, 0, :, :],Ux) + mul(G[1, 1, :, :],Uy)

    W = Wx @ Dh + Dh.T @ Wy

    return mul(Rmask,W)


def interp_setup(Nf,Nc):    #  interp mat
    zf, wf = gauss_lobatto(Nf)
    zc, wc = gauss_lobatto(Nc)
    Jh = interp_mat(zf,zc)
    return Jh

def gen_operators(N,X,Y):
    z, w, Dh = semhat(N)
    Bh = np.diag(w)

    w = w.reshape((N+1,1))
    Bxy= w @ w.T

    Ah = Dh.T @ Bh @ Dh
    Ah = 0.5 * (Ah + Ah.T)

    G, J, B = geometric_factors_2d(X, Y, w, Dh, N+1)
    return Ah, Bh, Dh, Bxy, G, J

def coarse_operators_setup(Ah, Bh, Jh_c2f, Jh_f2c=None):  # get coarse operators
    if Jh_f2c is None:
        Jh_f2c = Jh_c2f.T
    Ahc = Jh_f2c @ Ah @ Jh_c2f
    Bhc = Jh_f2c @ Bh @ Jh_c2f
    return Ahc, Bhc 

def fdm_2d_setup(Ahc, Bhc, Rxc, Ryc, Rmaskc):  # prepare for fdm
    Axc = Rxc @ Ahc @ Rxc.T
    Ayc = Ryc @ Ahc @ Ryc.T
    Bxc = Rxc @ Bhc @ Rxc.T 
    Byc = Ryc @ Bhc @ Ryc.T

    Lxc, Sxc = sla.eigh(Axc, Bxc)
    Lyc, Syc = sla.eigh(Ayc, Byc)
    
    __SRxc = Rxc.T @ Sxc
    __SRyc = Ryc.T @ Syc
    __SRxc = __SRxc.real
    __SRyc = __SRyc.real
      
    Lxc = Lxc.real
    Lyc = Lyc.real
    Lxc = Lxc.reshape((Lxc.shape[0],1))
    Lyc = Lyc.reshape((Lyc.shape[0],1)) 
    exc = 1+0*Lxc
    eyc = 1+0*Lyc
    Dc =  Lyc @ exc.T + eyc @ Lxc.T
    __dcinv = 1.0 / Dc
    
    __Rmaskc = Rmaskc
    return __SRxc, __SRyc, __dcinv

def fdm_2d_solve(U, SRx, SRy, dinv, Rmask):   # fdm as coarse grid solver
    U = mul(Rmask, U)
    U = SRy @ ( mul(dinv,(SRy.T @ U @ SRx)) ) @ SRx.T
    return mul(Rmask, U)

def hnorm(U, Bxy):
    Utmp = mul(U,U)
    return np.sqrt(np.sum(mul(Utmp,Bxy)))



def twolevels(funAx,b,set_mask2d,relax,relaxSetup
             ,msmth,Nf,Nc,tol=1e-8,maxit=100,x0=None,ivb=0,idumpu=False,crsmode=0):

    # Generate fine grid operators
    X, Y = reference_2d(Nf)
    Ah, Bh, Dh, Bxy, G, J = gen_operators(Nf,X,Y)
    Rx, Ry, Rmask = set_mask2d(Nf) # don't use function...

    # Generate coarse grid operators
    Jh_c2f  = interp_setup(Nf,Nc) # 2D coarse to fine J @ U @ J.T
    if crsmode<0:
        crsmode=abs(crsmode)
        Jh_f2c  = interp_setup(Nc,Nf)
    else:
        Jh_f2c = Jh_c2f.T

    Rxc, Ryc, Rmaskc = set_mask2d(Nc) # don't use function...

    if crsmode==1: # Interp everything in Galerkin sense
        Ahc, Bhc = coarse_operators_setup(Ah, Bh, Jh_c2f,Jh_f2c) # This gives full Bh (FIXME)

        # belows is only used when we need Ax at coarse grid
        # for box mesh + two levels, we have direct solves, no need Geometric factors
        # TODO: this is needed to coarse solve for non-box mesh
        Gc = np.zeros((2, 2, Nc+1, Nc+1)) 
        Gc [0, 0, :, :] = Jh_f2c @ G[0, 0, :, :] @ Jh_f2c.T 
        Gc [0, 1, :, :] = Jh_f2c @ G[0, 1, :, :] @ Jh_f2c.T
        Gc [1, 0, :, :] = Jh_f2c @ G[1, 0, :, :] @ Jh_f2c.T
        Gc [1, 1, :, :] = Jh_f2c @ G[1, 1, :, :] @ Jh_f2c.T
        Jc = Jh_f2c @ J @ Jh_f2c.T
 
    if crsmode==2: # Interp grid, gen matrix on grid
        Jh_f2c_tmp  = interp_setup(Nc,Nf) # for box mesh, this equals reference_2d(N)
        Xc = Jh_f2c_tmp @ X @ Jh_f2c_tmp.T # only used for setup
        Yc = Jh_f2c_tmp @ Y @ Jh_f2c_tmp.T
#       Xc, Yc = reference_2d(Nc)
        Ahc, Bhc, Dhc, Bxyc, Gc, Jc = gen_operators(Nc,Xc,Yc) # This gives diag Bh

    # Set up coarse grid solver (TODO, fdm for box mesh only)
    SRxc, SRyc, dcinv = fdm_2d_setup(Ahc, Bhc, Rxc, Ryc, Rmaskc)


    def funAx_interface_f(U): # to support different grids
        return fun_MG_Ax(U,Dh,G,Rmask)
    funAx = funAx_interface_f

    relaxData = relaxSetup(funAx,X.shape)

    timer0 = tic()

    if x0 is None:
        x = 0 * b
        res = hnorm(b, Bxy)
    else:
        x = x0 
        res = hnorm(b - funAx(x), Bxy)

    res_list = [res]

    ulist=[]
    if(idumpu): ulist=[x] # to plot error
    if(ivb==1): print("res[0] = %g"%res_list[-1])

    niter = 0; TOL = max(tol * res, tol)
    while niter < maxit and res > TOL:
        niter = niter + 1

        x = relax(funAx,x,b,msmth,**relaxData)    # relaxation
        r = b - funAx(x)

        rc = Jh_f2c @ r @ Jh_f2c.T                       # fine to coarse
        ec = fdm_2d_solve(rc, SRxc, SRyc, dcinv, Rmaskc) # coarse solve
        ef = Jh_c2f @ ec @ Jh_c2f.T                      # coarse to fine

        x = x + ef            # update

        res = hnorm(b - funAx(x), Bxy)
        res_list.append(res)
        if(idumpu): ulist.append(x)
        if(ivb==1): print("res[%d] = %g"%(niter,res_list[-1]))

    timer1 = toc(timer0)

    return x,niter,res_list,ulist,timer1


def threelevels(funAx,b,set_mask2d,relax,relaxSetup
               ,msmth,Nf,Nc1,Nc2
               ,tol=1e-8,maxit=100,x0=None,ivb=0,idumpu=False,crsmode=0):

    # Generate fine grid operators
    X, Y = reference_2d(Nf)
    Ah, Bh, Dh, Bxy, G, J = gen_operators(Nf,X,Y)
    Rx, Ry, Rmask = set_mask2d(Nf) # don't use function...

    # Generate coarse grid operators
    Jh1_c2f  = interp_setup(Nf, Nc1)
    Rxc1, Ryc1, Rmaskc1 = set_mask2d(Nc1) # don't use function...

    Jh2_c2f  = interp_setup(Nc1,Nc2)
    Rxc2, Ryc2, Rmaskc2 = set_mask2d(Nc2) # don't use function...

    if crsmode<0:
        crsmode=abs(crsmode)
        Jh1_f2c  = interp_setup(Nc1,Nf)
        Jh2_f2c  = interp_setup(Nc2,Nc1)
    else: # symmetric
        Jh1_f2c = Jh1_c2f.T
        Jh2_f2c = Jh2_c2f.T

    if crsmode==1: # Interp everything in Galerkin sense
        Ahc1, Bhc1 = coarse_operators_setup(Ah,  Bh,  Jh1_c2f,Jh1_f2c) # This full Bh (FIXME)
        Ahc2, Bhc2 = coarse_operators_setup(Ahc1,Bhc1,Jh2_c2f,Jh2_f2c) 
        _, _, Dhc1 = semhat(Nc1)
        _, _, Dhc2 = semhat(Nc2)

        # belows is only used when we need Ax at coarse grid
        Gc1 = np.zeros((2, 2, Nc1+1, Nc1+1)) 
        Gc1 [0, 0, :, :] = Jh1_f2c @ G[0, 0, :, :] @ Jh1_f2c.T
        Gc1 [0, 1, :, :] = Jh1_f2c @ G[0, 1, :, :] @ Jh1_f2c.T
        Gc1 [1, 0, :, :] = Jh1_f2c @ G[1, 0, :, :] @ Jh1_f2c.T
        Gc1 [1, 1, :, :] = Jh1_f2c @ G[1, 1, :, :] @ Jh1_f2c.T
        Jc1 = Jh1_f2c @ J @ Jh1_f2c.T

        Gc2 = np.zeros((2, 2, Nc2+1, Nc2+1))
        Gc2 [0, 0, :, :] = Jh2_f2c @ Gc1[0, 0, :, :] @ Jh2_f2c.T
        Gc2 [0, 1, :, :] = Jh2_f2c @ Gc1[0, 1, :, :] @ Jh2_f2c.T
        Gc2 [1, 0, :, :] = Jh2_f2c @ Gc1[1, 0, :, :] @ Jh2_f2c.T
        Gc2 [1, 1, :, :] = Jh2_f2c @ Gc1[1, 1, :, :] @ Jh2_f2c.T
        Jc2 = Jh2_f2c @ Jc1 @ Jh2_f2c.T
 
    if crsmode==2: # Interp grid, gen matrix on grid
        Jh1_f2c_tmp  = interp_setup(Nc1,Nf) # for box mesh, this equals reference_2d(N)
        Xc1 = Jh1_f2c_tmp @ X @ Jh1_f2c_tmp.T # only used for setup
        Yc1 = Jh1_f2c_tmp @ Y @ Jh1_f2c_tmp.T
#       Xc1, Yc1 = reference_2d(Nc1)

        Jh2_f2c_tmp  = interp_setup(Nc2,Nc1) 
        Xc2 = Jh2_f2c_tmp @ Xc1 @ Jh2_f2c_tmp.T 
        Yc2 = Jh2_f2c_tmp @ Yc1 @ Jh2_f2c_tmp.T
#       Xc2, Yc2 = reference_2d(Nc2)

        Ahc1, Bhc1, Dhc1, Bxyc1, Gc1, Jc1 = gen_operators(Nc1,Xc1,Yc1) # This gives diag Bh
        Ahc2, Bhc2, Dhc2, Bxyc2, Gc2, Jc2 = gen_operators(Nc2,Xc2,Yc2) # This gives diag Bh

    # Set up coarse grid solver (TODO, fdm for box mesh only)
    SRxc2, SRyc2, dcinv2 = fdm_2d_setup(Ahc2, Bhc2, Rxc2, Ryc2, Rmaskc2)

    # to support different grids, TODO, need a class
    def funAx_interface_f(U):
        return fun_MG_Ax(U,Dh,G,Rmask)
    def funAx_interface_c1(Uc1): 
        return fun_MG_Ax(Uc1,Dhc1,Gc1,Rmaskc1)
    def funAx_interface_c2(Uc2): 
        return fun_MG_Ax(Uc2,Dhc2,Gc2,Rmaskc2)

    funAx_f  = funAx_interface_f
    funAx_c1 = funAx_interface_c1
    funAx_c2 = funAx_interface_c2

    relaxData_f  = relaxSetup(funAx_f, X.shape)
    relaxData_c1 = relaxSetup(funAx_c1,(Nc1+1,Nc1+1))
    relaxData_c2 = relaxSetup(funAx_c2,(Nc2+1,Nc2+1))

    timer0 = tic() # exclude setup time

    if x0 is None:
        x = 0 * b
        res = hnorm(b, Bxy)
    else:
        x = x0 
        res = hnorm(b - funAx(x), Bxy)

    res_list = [res]

    ulist=[]
    if(idumpu): ulist=[x]
    if(ivb==1): print("res[0] = %g"%res_list[-1])

    uf = x
    bf = b

    niter = 0; TOL = max(tol * res, tol)
    while niter < maxit and res > TOL:
        niter = niter + 1                                     #F   #C1  #C2

        uf  = relax(funAx_f,uf,bf,msmth,**relaxData_f)        # relaxation
        bc1 = Jh1_f2c @ (bf - funAx_f(uf)) @ Jh1_f2c.T        # fine to coarse

        uc1 = 0*bc1
        uc1 = relax(funAx_c1,uc1,bc1,msmth,**relaxData_c1)         # relaxation
        bc2 = Jh2_f2c @ (bc1 - funAx_c1(uc1)) @ Jh2_f2c.T          # fine to coarse

        uc2 = 0*bc2
        uc2 = fdm_2d_solve(bc2, SRxc2, SRyc2, dcinv2, Rmaskc2)          # coarse solve

        uc1 = uc1 + Jh2_c2f @ uc2 @ Jh2_c2f.T                      # coarse to fine
        uc1 = relax(funAx_c1,uc1,bc1,msmth,**relaxData_c1)         # relaxation

        uf  = uf + Jh1_c2f @ uc1 @ Jh1_c2f.T                  # coarse to fine
        uf  = relax(funAx_f,uf,bf,msmth,**relaxData_f)        # relaxation

        res = hnorm(bf - funAx_f(uf), Bxy)
        res_list.append(res)
        if(idumpu): ulist.append(uf)
        if(ivb==1): print("res[%d] = %g"%(niter,res_list[-1]))
    
    timer1 = toc(timer0)

    return uf,niter,res_list,ulist,timer1



