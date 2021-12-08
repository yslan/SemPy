import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from sempy.meshes.curved import trapezoid
from sempy.meshes.box import reference, reference_2d, box_ab

from sempy.gradient import (
    gradient,
    gradient_2d,
    gradient_transpose,
    gradient_transpose_2d,
)

from sempy.iterative import cg, pcg

from sempy.derivative import reference_derivative_matrix

from sempy.mass import reference_mass_matrix_1d

from mayavi import mlab
import matplotlib.pyplot as plt

import time

example_2d = 1
plot_on = 1

elapsed_cg = []
elapsed_fdm = []
niters_cg = []
niters_fdm = []
orders = []

#    mesh = load_mesh("box001.msh")
#    mesh.find_physical_coordinates(N)
#    mesh.calc_geometric_factors()
#    mesh.establish_global_numbering()
#    mesh.setup_mask()
#
#    G = mesh.get_geom()
#    J = mesh.get_jaco()
#    B = mesh.get_mass()

def geometric_factors(X,Y,Z,n):
    from sempy.mass import reference_mass_matrix_3d, reference_mass_matrix_2d
    xr, xs, xt = gradient(X, n)
    yr, ys, yt = gradient(Y, n)
    zr, zs, zt = gradient(Z, n)

    J = (
        xr * (ys * zt - yt * zs)
        - yr * (xs * zt - xt * zs)
        + zr * (xs * yt - ys * xt)
    )

    rx = (ys * zt - yt * zs) / J
    sx = (yt * zr - yr * zt) / J
    tx = (yr * zs - ys * zr) / J

    ry = -(zt * xs - zs * xt) / J
    sy = -(zr * xt - zt * xr) / J
    ty = -(zs * xr - zr * xs) / J

    rz = (xs * yt - xt * ys) / J
    sz = -(xr * yt - xt * yr) / J
    tz = (xr * ys - xs * yr) / J

    g11 = rx * rx + ry * ry + rz * rz
    g12 = rx * sx + ry * sy + rz * sz
    g13 = rx * tx + ry * ty + rz * tz
    g22 = sx * sx + sy * sy + sz * sz
    g23 = sx * tx + sy * ty + sz * tz
    g33 = tx * tx + ty * ty + tz * tz

    B = reference_mass_matrix_3d(n - 1)

    g = np.zeros((3, 3, g11.size))
    g[0, 0, :] = g11 * B * J
    g[0, 1, :] = g12 * B * J
    g[0, 2, :] = g13 * B * J
    g[1, 0, :] = g12 * B * J
    g[1, 1, :] = g22 * B * J
    g[1, 2, :] = g23 * B * J
    g[2, 0, :] = g13 * B * J
    g[2, 1, :] = g23 * B * J
    g[2, 2, :] = g33 * B * J
    g = np.array(g)
    J = np.array(J)
    B = np.array(B)
    return g,J,B
def geometric_factors_2d(X,Y,n):
    from sempy.mass import reference_mass_matrix_3d, reference_mass_matrix_2d
    Xr, xs = gradient_2d(X, n)
    yr, ys = gradient_2d(Y, n)

    J = Xr * ys - yr * xs

    rx = ys / J
    sx = -yr / J

    ry = -xs / J
    sy = Xr / J

    g11 = rx * rx + ry * ry
    g12 = rx * sx + ry * sy
    g22 = sx * sx + sy * sy

    B = reference_mass_matrix_2d(n - 1)

    g = np.zeros((2, 2, g11.size))
    g[0, 0, :] = g11 * B * J
    g[0, 1, :] = g12 * B * J
    g[1, 0, :] = g12 * B * J
    g[1, 1, :] = g22 * B * J
    g = np.array(g)
    J = np.array(J)
    B = np.array(B)
    return g,J,B


for N in range(2, 10):
    n = N + 1

    if example_2d:
        X, Y = reference_2d(N)
        G, J, B = geometric_factors_2d(X, Y, n)
    else:
        X, Y, Z = trapezoid(N)
        G, J, B = geometric_factors(X, Y, Z, n)

    def fdm_d_inv_2d(p):
        n = p + 1

        Bh = reference_mass_matrix_1d(p)
        Dh = reference_derivative_matrix(p)
        Ah = Dh.T @ Bh @ Dh
        Ah = 0.5 * (Ah + Ah.T)

        I = np.identity(n, dtype=np.float64)
        Rx = I
        Ry = I[1:, :]

        Ax = Rx @ Ah @ Rx.T
        nx = Ax.shape[0]
        Ay = Ry @ Ah @ Ry.T
        ny = Ay.shape[0]
        Bx = Rx @ Bh @ Rx.T
        By = Ry @ Bh @ Ry.T

        Lx, Sx = sla.eigh(Ax, Bx)
        Lx = np.diag(Lx)
        Ix = np.identity(nx, dtype=np.float64)
        Ly, Sy = sla.eigh(Ay, By)
        Ly = np.diag(Ly)
        Iy = np.identity(ny, dtype=np.float64)

        Lx = Lx.real
        Ly = Ly.real

        D = np.kron(Iy, Lx) + np.kron(Ly, Ix)
        dinv = 1.0 / np.diag(D)

        return Rx, Ry, Sx.real, Sy.real, dinv

    def fdm_d_inv(p):
        n = p + 1

        Bh = reference_mass_matrix_1d(p)
        Dh = reference_derivative_matrix(p)
        Ah = Dh.T @ Bh @ Dh
        Ah = 0.5 * (Ah + Ah.T)

        I = np.identity(n, dtype=np.float64)
        Rx = Rz = I
        Ry = I[1:, :]

        Ax = Rx @ Ah @ Rx.T
        nx = Ax.shape[0]
        Ay = Ry @ Ah @ Ry.T
        ny = Ay.shape[0]
        Az = Rz @ Ah @ Rz.T
        nz = Az.shape[0]
        Bx = Rx @ Bh @ Rx.T
        By = Ry @ Bh @ Ry.T
        Bz = Rz @ Bh @ Rz.T

        Lx, Sx = sla.eigh(Ax, Bx)
        Lx = np.diag(Lx)
        Ix = np.identity(nx, dtype=np.float64)
        Ly, Sy = sla.eigh(Ay, By)
        Ly = np.diag(Ly)
        Iy = np.identity(ny, dtype=np.float64)
        Lz, Sz = sla.eigh(Az, Bz)
        Lz = np.diag(Lz)
        Iz = np.identity(nz, dtype=np.float64)

        Lx = Lx.real
        Ly = Ly.real
        Lz = Lz.real

        D = (
            np.kron(Iz, np.kron(Iy, Lx))
            + np.kron(Iz, np.kron(Ly, Ix))
            + np.kron(Lz, np.kron(Iy, Ix))
        )
        dinv = 1.0 / np.diag(D)

        return Rx, Ry, Rz, Sx.real, Sy.real, Sz.real, dinv

    def mask(W):
        return np.dot(R.T, np.dot(R, W))

    def Ax_2d(x):
        Ux, Uy = gradient_2d(x, n)

        Wx = G[0, 0, :] * Ux + G[0, 1, :] * Uy
        Wy = G[1, 0, :] * Ux + G[1, 1, :] * Uy

        W = gradient_transpose_2d(Wx, Wy, n)
        return mask(W)

    def Ax(x):
        Ux, Uy, Uz = gradient(x, n)

        Wx = G[0, 0, :] * Ux + G[0, 1, :] * Uy + G[0, 2, :] * Uz
        Wy = G[1, 0, :] * Ux + G[1, 1, :] * Uy + G[1, 2, :] * Uz
        Wz = G[2, 0, :] * Ux + G[2, 1, :] * Uy + G[2, 2, :] * Uz

        W = gradient_transpose(Wx, Wy, Wz, n)
        return mask(W)

    Minv = 1.0 / (B * J)

    def precon_mass(r):
        return Minv * r

    def fast_kron_2d(Sy, Sx, U):
        nx, mx = Sx.shape
        ny, my = Sy.shape

        U = U.reshape((my, mx))
        U = np.dot(U, Sx.T)

        V = U.reshape((my, nx))
        U = np.dot(Sy, V)

        return U.reshape((nx * ny,))

    def fast_kron(Sz, Sy, Sx, U):
        nx, mx = Sx.shape
        ny, my = Sy.shape
        nz, mz = Sz.shape

        U = U.reshape((my * mz, mx))
        U = np.dot(U, Sx.T)

        U = U.reshape((mz, my, nx))
        V = np.zeros((mz, ny, nx))
        for i in range(mz):
            V[i, :, :] = np.dot(Sy, U[i, :, :])

        V = V.reshape((mz, nx * ny))
        U = np.dot(Sz, V)

        return U.reshape((nx * ny * nz,))

    if example_2d:
        Rx, Ry, Sx, Sy, dinv = fdm_d_inv_2d(N)
        R = np.kron(Ry, Rx)
    else:
        Rx, Ry, Rz, Sx, Sy, Sz, dinv = fdm_d_inv(N)
        R = np.kron(Rz, np.kron(Ry, Rx))

    def precon_fdm_2d(r):
        r = np.dot(R, r)
        b = fast_kron_2d(Sy.T, Sx.T, r)
        b = dinv * b
        return np.dot(R.T, fast_kron_2d(Sy, Sx, b))

    def precon_fdm(r):
        r = np.dot(R, r)
        b = fast_kron(Sz.T, Sy.T, Sx.T, r)
        b = dinv * b
        return np.dot(R.T, fast_kron(Sz, Sy, Sx, b))

    def precon_jacobi_2d(r):
        r = np.dot(R, r)
        b = fast_kron_2d(Sy.T, Sx.T, r)
        b = dinv * b
        return np.dot(R.T, fast_kron_2d(Sy, Sx, b))

    def precon_jacobi(r):
        r = np.dot(R, r)
        b = fast_kron(Sz.T, Sy.T, Sx.T, r)
        b = dinv * b
        return np.dot(R.T, fast_kron(Sz, Sy, Sx, b))

    if example_2d:
        b = np.exp(10 * Y) * np.sin(10 * X)
        b = mask(b.reshape((n * n,)) * B * J)
    else:
        b = np.exp(10 * Y * Z) * np.sin(10 * X)
        b = mask(b.reshape((n * n * n,)) * B * J)

    tol = 1.0e-8
    maxit = 1000
    verbose = 0

    t = time.process_time()

    if example_2d:
        x_cg, niter_cg = cg(Ax_2d, b, tol, maxit, verbose)
        tt = time.process_time() - t
        elapsed_cg.append(tt)
        x_mass, niter_mass = pcg(Ax_2d, precon_mass, b, tol, maxit, verbose)
        x_jacobi, niter_jacobi = pcg(Ax_2d, precon_jacobi_2d, b, tol, maxit, verbose)
        t = time.process_time()
        x_fdm, niter_fdm = pcg(Ax_2d, precon_fdm_2d, b, tol, maxit, verbose)
        tt = time.process_time() - t
        elapsed_fdm.append(tt)
    else:
        x_cg, niter_cg = cg(Ax, b, tol, maxit, verbose)
        tt = time.process_time() - t
        elapsed_cg.append(tt)
        #    x_mass  ,niter_mass  =pcg(Ax,precon_mass  ,b,tol,maxit,verbose)
        #    x_jacobi,niter_jacobi=pcg(Ax,precon_jacobi,b,tol,maxit,verbose)
        t = time.process_time()
        x_fdm, niter_fdm = pcg(Ax, precon_fdm, b, tol, maxit, verbose)
        tt = time.process_time() - t
        elapsed_fdm.append(tt)

    niters_fdm.append(niter_fdm)
    niters_cg.append(niter_cg)
    orders.append(N)

plt.figure()
plt.plot(orders, elapsed_cg, "-o")
plt.title("Order vs Elapsed time for CG", fontsize=20)
plt.xlim(1, N + 1)
plt.xlabel("N - order", fontsize=16)
plt.ylabel("time (s)", fontsize=16)
plt.savefig("elapsed_cg.pdf", bbox_inches="tight")

plt.figure()
plt.plot(orders, niters_cg, "-o")
plt.title("Order vs number of iterations for CG", fontsize=20)
plt.xlim(2, N + 1)
plt.ylabel("# iterations", fontsize=16)
plt.xlabel("N - order", fontsize=16)
plt.savefig("niter_cg.pdf", bbox_inches="tight")

plt.figure()
plt.semilogy(orders, elapsed_fdm, "b-o", label="pcg(fdm)")
plt.semilogy(orders, elapsed_cg, "g-o", label="cg")
plt.title("Order vs Elapsed time", fontsize=20)
plt.xlim(1, N + 1)
plt.xlabel("N - order", fontsize=16)
plt.ylabel("time (s)", fontsize=16)
plt.legend(loc=0)
plt.savefig("elapsed_fdm_cg.pdf", bbox_inches="tight")

plt.figure()
plt.semilogy(orders, niters_fdm, "b-o", label="pcg(fdm)")
plt.semilogy(orders, niters_cg, "g-o", label="cg")
plt.title("Order vs number of iterations for FDM", fontsize=20)
plt.xlim(2, N + 1)
plt.ylabel("# iterations", fontsize=16)
plt.xlabel("N - order", fontsize=16)
plt.legend(loc=0)
plt.savefig("niter_fdm_cg.pdf", bbox_inches="tight")

if plot_on:
   if example_2d:
     print("N/A")
     plt.show()
   else:
       mlab.figure()
       mlab.points3d(X,Y,Z,(x_cg-x_fdm).reshape((n,n,n)),scale_mode="none",scale_factor=0.1)
       mlab.axes()
       mlab.show()
