from sempy.agg_amg.vcycle import vcycle

import numpy as np


def precond(r, A, J0, prec):
    n = A.shape[0]
    if prec == 0:
        Di = 1.0/np.diag(A)
        Di = Di.reshape((n, 1))
        z = np.multiply(Di, r)
        z = r.copy()
    elif prec == 1:
        z = vcycle(r, A, 0, J0)
    elif prec == 2:
        #z = kcycle(r, A, 0, J0)
        print("K-cycle is not implemented yet.")
        exit()
    else:
        z = r.copy()

    return z.reshape((n, 1))


def project(r, A, J0, tol, prec, verbose=1):
    n_iter, max_iter = 0, 1000

    if tol < 0:
        max_iter = abs(tol)
        tol = 1e-3

    n = r.shape[0]

    z = precond(r, A, J0, prec)
    rz1 = np.dot(r.T, z)[0, 0]

    x = np.zeros_like(r)
    p = z.copy()

    P = np.zeros((n, max_iter))
    W = np.zeros_like(P)

    res = []
    res.append(np.linalg.norm(r))

    if verbose > 0:
        print("A.shape: {} p.shape: {}".format(A.shape, p.shape))

    for k in range(max_iter):
        w = A.dot(p)
        pAp = np.dot(p.T, w)[0, 0]
        alpha = rz1/pAp

        if prec > 0:
            scale = 1./np.sqrt(pAp)
            W[:, k] = scale*w
            P[:, k] = scale*p

        if verbose > 0:
            print("x.shape: {}".format(x.shape))
        x = x+alpha*p
        r = r-alpha*w

        ek = np.linalg.norm(r)
        res.append(ek)

        if ek < tol:
            break

        zo = z.copy()
        z = precond(r, A, J0, prec)
        dz = z-zo

        rz0 = rz1
        rz1 = np.dot(r.T, z)[0, 0]
        rz2 = np.dot(r.T, dz)[0, 0]
        beta = rz2/rz0

        p = z+beta*p

        if prec > 0:
            a = np.dot(W[:, 0:k].T, p)
            p = p-np.dot(P[:, 0:k], a)
    return x, res, k
