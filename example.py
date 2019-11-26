import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from sempy.meshes.curved import trapezoid
from sempy.meshes.box import reference,box_ab

from sempy.stiffness import geometric_factors
from sempy.stiffness import gradient,gradient_transpose

from sempy.iterative import cg,pcg

from sempy.derivative import reference_derivative_matrix

from sempy.mass import reference_mass_matrix_1D

from mayavi import mlab
import matplotlib.pyplot as plt

N=10
n=N+1

X,Y,Z=reference(N)
G,J,B=geometric_factors(X,Y,Z,n)

restricted=1

def fdm_d_inv(p):
    n=p+1

    I=np.identity(n,dtype=np.float64)

    Bh=reference_mass_matrix_1D(p);
    Dh=reference_derivative_matrix(p)
    Ah=Dh.T@Bh@Dh; Ah=0.5*(Ah+Ah.T)

    Rx=Ry=Rz=I[1:-1,:]

    Ax=Rx@Ah@Rx.T; nx=Ax.shape[0]
    Ay=Ry@Ah@Ry.T; ny=Ay.shape[0]
    Az=Rz@Ah@Rz.T; nz=Az.shape[0]
    Bx=Rx@Bh@Rx.T
    By=Ry@Bh@Ry.T
    Bz=Rz@Bh@Rz.T

    Lx,Sx=sla.eig(Ax,Bx); Lx=np.diag(Lx); Ix=np.identity(nx,dtype=np.float64)
    Ly,Sy=sla.eig(Ay,By); Ly=np.diag(Ly); Iy=np.identity(ny,dtype=np.float64)
    Lz,Sz=sla.eig(Az,Bz); Lz=np.diag(Lz); Iz=np.identity(nz,dtype=np.float64)

    Lx=Lx.real
    Ly=Ly.real
    Lz=Lz.real

    if not restricted:
        Rx=Ry=Rz=I

    D=np.kron(Iz,np.kron(Iy,Lz))+np.kron(Iz,np.kron(Ly,Ix))+np.kron(Lx,np.kron(Iy,Ix))
    dinv=1.0/np.diag(D)
    dinv[dinv>10]=0.0

    return Rx,Ry,Rz,Sx.real,Sy.real,Sz.real,dinv

Rx,Ry,Rz,Sx,Sy,Sz,dinv=fdm_d_inv(N)
R=np.kron(Rz,np.kron(Ry,Rx))
print("min: {}".format(np.min(np.abs(dinv))))
print("max: {}".format(np.max(np.abs(dinv))))

def mask(W):
    W=W.reshape((n*n*n,))
    return np.dot(R.T,np.dot(R,W))

def Ax(x):
    Ux,Uy,Uz=gradient(x,n)

    Wx=G[0,0,:]*Ux+G[0,1,:]*Uy+G[0,2,:]*Uz
    Wy=G[1,0,:]*Ux+G[1,1,:]*Uy+G[1,2,:]*Uz
    Wz=G[2,0,:]*Ux+G[2,1,:]*Uy+G[2,2,:]*Uz

    W=gradient_transpose(Wx,Wy,Wz,n)
    W=mask(W)
    return W

Minv=1.0/(B*J)
def precon_mass(r):
    return Minv*r


def fast_kron(Sz,Sy,Sx,U):
    nx,mx=Sx.shape
    ny,my=Sy.shape
    nz,mz=Sz.shape

    U=U.reshape((my*mz,mx))
    U=np.dot(U,Sx.T)

    U=U.reshape((mz,my,nx))
    V=np.zeros ((mz,ny,nx))
    for i in range(mz):
        V[i,:,:]=np.dot(Sy,U[i,:,:])

    V=V.reshape((mz,nx*ny))
    U=np.dot(Sz,V)

    return U.reshape((nx*ny*nz,))

def precon_fdm(r):
    r=np.dot(R,r)
    b=fast_kron(Sz.T,Sy.T,Sx.T,r)
    b=dinv*b
    return np.dot(R.T,fast_kron(Sz,Sy,Sx,b))

b=np.exp(10*Y*Z)*np.sin(10*X)
b=mask(b.reshape((n*n*n,))*B*J)

tol=1.e-10
maxit=10000
verbose=0

x_cg  ,niter_cg  =cg (Ax,            b,tol,maxit,verbose)
x_mass,niter_mass=pcg(Ax,precon_mass,b,tol,maxit,verbose)
x_fdm ,niter_fdm =pcg(Ax,precon_fdm ,b,tol,maxit,verbose)
print("# iterations: cg {} pcg (mass) {} pcg (fdm) {}".format(niter_cg,
  niter_mass,niter_fdm))

print("error: {}".format(np.max(np.abs(x_cg-x_fdm))))
plot=1
if plot:
    mlab.figure()
    mlab.points3d(X,Y,Z,(x_cg-x_fdm).reshape((n,n,n)),scale_mode="none",scale_factor=0.1)
    mlab.axes()
    mlab.show()
