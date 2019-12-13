import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from sempy.mesh import load_mesh

from sempy.stiffness import gradient,gradient_2d,\
    gradient_transpose,gradient_transpose_2d

from sempy.iterative import cg,pcg

from mayavi import mlab
import matplotlib.pyplot as plt

N=5
n=N+1

mesh=load_mesh("box002.msh")
mesh.find_physical_nodes(N)
mesh.calc_geometric_factors()

print(mesh.jaco[0,:])
X=mesh.xe
Y=mesh.ye
Z=mesh.ze

print("{} {} {}".format(X.shape,Y.shape,Z.shape))

example_2d=0
plot_on   =1

if plot_on:
    if example_2d:
      print("N/A")
    else:
        mlab.figure()
        mlab.points3d(X,Y,Z,X,\
            scale_mode="none",scale_factor=0.1)
        mlab.axes()
        mlab.show()
