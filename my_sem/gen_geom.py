import numpy as np
from my_sem.gen_semhat import reference_derivative_matrix, gauss_lobatto # TODO

'''
    Generate the geometric factors, jacobian, and mass
'''


def geometric_factors_2d_arr(xx,yy,n):
    assert xx.ndim == 1
    assert yy.ndim == 1

    N = n-1
    nn = n*n

    Dh = reference_derivative_matrix(N)

    X = xx.reshape(n, n)
    Xr = np.dot(X, Dh.T)
    Xs = np.dot(Dh, X)

    Y = yy.reshape(n, n)
    Yr = np.dot(Y, Dh.T)
    Ys = np.dot(Dh, Y)

    xxr = Xr.reshape((nn,))
    xxs = Xs.reshape((nn,))
    yyr = Yr.reshape((nn,))
    yys = Ys.reshape((nn,))

    J = xxr * yys - yyr * xxs

    rxx = yys / J
    sxx = -yyr / J

    ryy = -xxs / J
    syy = xxr / J

    g11 = rxx * rxx + ryy * ryy
    g12 = rxx * sxx + ryy * syy
    g22 = sxx * sxx + syy * syy

    z, w = gauss_lobatto(N) 
    w = w.reshape((n,1))
    Bxy = w @ w.T
    Bhh = Bxy.reshape((nn,))

    G = np.zeros((2, 2, g11.size))
    G[0, 0, :] = g11 * Bhh * J
    G[0, 1, :] = g12 * Bhh * J
    G[1, 0, :] = g12 * Bhh * J
    G[1, 1, :] = g22 * Bhh * J
    G  = np.array(G)
    J  = np.array(J)
    Bhh= np.array(Bhh)
    return G, J, Bhh


def geometric_factors_2d(X,Y,w,Dh,n):
    assert X.ndim == 2
    assert Y.ndim == 2

    N = n-1
    nn = n*n

    Xr = np.dot(X, Dh.T)
    Xs = np.dot(Dh, X)

    Yr = np.dot(Y, Dh.T)
    Ys = np.dot(Dh, Y)

    J = Xr * Ys - Yr * Xs

    rX = Ys / J
    sX = -Yr / J

    rY = -Xs / J
    sY = Xr / J

    g11 = rX * rX + rY * rY
    g12 = rX * sX + rY * sY
    g22 = sX * sX + sY * sY

    w = w.reshape((n,1))
    Bxy= w @ w.T

    G = np.zeros((2, 2)+g11.shape)
    G[0, 0, :, :] = g11 * Bxy * J
    G[0, 1, :, :] = g12 * Bxy * J
    G[1, 0, :, :] = g12 * Bxy * J
    G[1, 1, :, :] = g22 * Bxy * J
    G  = np.array(G)
    J  = np.array(J)
    Bxy= np.array(Bxy)
    return G, J, Bxy


