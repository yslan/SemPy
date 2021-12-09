import numpy as np
from my_sem.gen_semhat import reference_derivative_matrix, gauss_lobatto # TODO
from numpy import multiply as mul

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
    Xr = X @ Dh.T
    Xs = Dh @ X

    Y = yy.reshape(n, n)
    Yr = Y @ Dh.T
    Ys = Dh @ Y

    xxr = Xr.reshape((nn,))
    xxs = Xs.reshape((nn,))
    yyr = Yr.reshape((nn,))
    yys = Ys.reshape((nn,))

    J = mul(xxr,yys) - mul(yyr,xxs)

    rxx = yys / J
    sxx = -yyr / J

    ryy = -xxs / J
    syy = xxr / J

    g11 = mul(rxx,rxx) + mul(ryy,ryy)
    g12 = mul(rxx,sxx) + mul(ryy,syy)
    g22 = mul(sxx,sxx) + mul(syy,syy)

    z, w = gauss_lobatto(N) 
    w = w.reshape((n,1))
    Bxy = w @ w.T
    Bhh = Bxy.reshape((nn,))

    G = np.zeros((2, 2, g11.size))
    G[0, 0, :] = mul(g11, mul(Bhh,J) )  
    G[0, 1, :] = mul(g12, mul(Bhh,J) )
    G[1, 0, :] = mul(g12, mul(Bhh,J) )
    G[1, 1, :] = mul(g22, mul(Bhh,J) )
    G  = np.array(G)
    J  = np.array(J)
    Bhh= np.array(Bhh)
    return G, J, Bhh


def geometric_factors_2d(X,Y,w,Dh,n):
    assert X.ndim == 2
    assert Y.ndim == 2

    N = n-1
    nn = n*n

    Xr = X @ Dh.T
    Xs = Dh @ X

    Yr = Y @ Dh.T
    Ys = Dh @ Y

    J = mul(Xr,Ys) - mul(Yr,Xs)

    rX = Ys / J
    sX = -Yr / J

    rY = -Xs / J
    sY = Xr / J

    g11 = mul(rX,rX) + mul(rY,rY)
    g12 = mul(rX,sX) + mul(rY,sY)
    g22 = mul(sX,sX) + mul(sY,sY)

    w = w.reshape((n,1))
    Bxy= w @ w.T

    G = np.zeros((2, 2)+g11.shape)
    G[0, 0, :, :] = mul(g11, mul(Bxy,J) ) 
    G[0, 1, :, :] = mul(g12, mul(Bxy,J) )
    G[1, 0, :, :] = mul(g12, mul(Bxy,J) )
    G[1, 1, :, :] = mul(g22, mul(Bxy,J) )
    G  = np.array(G)
    J  = np.array(J)
    Bxy= np.array(Bxy)
    return G, J, Bxy


