import numpy as np

from sempy.meshes.curved import trapezoid
from sempy.meshes.box import reference,box_ab
from sempy.stiffness import geometric_factors
from sempy.derivative import reference_gradient,reference_gradient_transpose
from sempy.quadrature import gauss_lobatto
from sempy.iterative import cg,pcg

from mayavi import mlab

N=10
n=N+1

X,Y,Z=box_ab(-2.,2.,N)
G,J,B=geometric_factors(X,Y,Z,n)

def mask(W):
    W=W.reshape((n,n,n))
    W[0,:,:]=0
    W[n-1,:,:]=0
    W[:,0,:]=0
    W[:,n-1,:]=0
    W[:,:,0]=0
    W[:,:,n-1]=0
    W=W.reshape((n*n*n,))
    return W

def Ax(x):
    Ux,Uy,Uz=reference_gradient(x,n)

    Wx=G[0,0,:]*Ux+G[0,1,:]*Uy+G[0,2,:]*Uz
    Wy=G[1,0,:]*Ux+G[1,1,:]*Uy+G[1,2,:]*Uz
    Wz=G[2,0,:]*Ux+G[2,1,:]*Uy+G[2,2,:]*Uz

    W=reference_gradient_transpose(Wx,Wy,Wz,n)
    W=mask(W)
    return W

x_analytic=np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
x_analytic=x_analytic.reshape((n*n*n,))
x_analytic=mask(x_analytic)

b=Ax(x_analytic)
b=mask(b)

b0=3*np.pi*np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
b0=b0.reshape((n*n*n,))
b0=b0*B
b0=mask(b0)

x,niter=cg(Ax,b0,tol=1e-12,maxit=1000,verbose=1)
#Minv=1.0/(B)
#x,niter=pcg(Ax,Minv,b0,tol=1e-12,maxit=1000,verbose=1)
print(np.max(np.abs(x-x_analytic)))

mlab.figure()
mlab.points3d(X,Y,Z,(x-x_analytic).reshape((n,n,n)),scale_mode="none",scale_factor=0.1)
mlab.axes()
mlab.show()
