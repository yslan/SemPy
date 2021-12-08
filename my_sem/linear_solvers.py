'''
    Linear solvers
      CG
      GMRES
    Inputs:
      fun_A
      fun_Minv
      b
      x0
      wt
      tol
      maxit
      verbose
'''
import numpy as np


def my_dot(u,v,wt):
    return np.sum(u*v*wt)

def cg(A, b, x0=None, wt=None, tol=1e-12, maxit=100, verbose=0):
    if wt is None:
      wt = 0 * b + 1

    if x0 is None:
      x = 0 * b
      r = b
      rdotr = my_dot(r, r, wt)
    else:
      x = x0
      r = A(x)
      rdotr = my_dot(r, r, wt)

    if verbose:
        print("Initial rnorm={}".format(rdotr))

    TOL = max(tol * tol * rdotr, tol * tol)


    niter = 0
    if rdotr < 1.0e-20:
        return x, niter

    p = r
    while niter < maxit and rdotr > TOL:
        Ap = A(p)
        pAp = my_dot(p, Ap, wt)

        alpha = rdotr / pAp

        x = x + alpha * p
        r = r - alpha * Ap

        rdotr0 = rdotr
        rdotr = my_dot(r, r, wt)
        beta = rdotr / rdotr0

        if verbose:
            print(
                "niter={} r0={} r1={} alpha={} beta={} pap={}".format(
                    niter, rdotr0, rdotr, alpha, beta, pAp
                )
            )

        p = r + beta * p
        niter = niter + 1

    return x, niter


def pcg(A, Minv, b, x0=None, wt=None, tol=1e-8, maxit=100, verbose=0):
    if wt is None:
      wt = 0 * b + 1

    if x0 is None:
      x = 0 * b
      r = b
      rdotr = my_dot(r, r, wt)
    else:
      x = x0
      r = A(x)
      rdotr = my_dot(r, r, wt)

    if verbose:
        print("Initial rnorm={}".format(rdotr))

    TOL = max(tol * tol * rdotr, tol * tol)


    niter = 0

    z = Minv(r)
    rdotz = my_dot(r, z, wt)

    p = z
    while niter < maxit and rdotz > TOL:
        Ap = A(p)
        pAp = my_dot(p, Ap, wt)

        alpha = rdotz / pAp

        x = x + alpha * p
        r = r - alpha * Ap

        z = Minv(r)

        rdotz0 = rdotz
        rdotz = my_dot(r, z, wt)
        beta = rdotz / rdotz0
        if verbose:
            print(
                "niter={} r0={} r1={} alpha={} beta={} pap={}".format(
                    niter, rdotz0, rdotz, alpha, beta, pAp
                )
            )

        p = z + beta * p
        niter = niter + 1

    return x, niter

