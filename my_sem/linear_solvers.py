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
from numpy import multiply as mul


def my_dot(u,v,wt):
    return np.sum(mul(mul(u,v),wt))

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

    TOL = max(tol * tol * rdotr, tol * tol)

    niter = 0

    z = Minv(r)
    rdotz = my_dot(r, z, wt)

    if verbose:
        print("Initial rnorm={}".format(rdotr),rdotz,TOL)
        if(rdotz<0): # cheb-mass fails
           quit()

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


def arnoldi_iteration(funA, b, n: int): # copy from wiki
    """Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^n b}.

    Arguments
      A: m × m array
      b: initial vector (length m)
      n: dimension of Krylov subspace, must be >= 1
    
    Returns
      Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
      h: (n + 1) x n array, A on basis Q. It is upper Hessenberg.  
    """
    eps = 1e-12
    h = np.zeros((n+1,n))
    m = b.size; mm = b.shape
    Q = np.zeros((m,n+1))
    # Normalize the input vector
    Qtmp = b/np.linalg.norm(b,2)   # Use it as the first Krylov vector
    Q[:,0] = Qtmp.reshape((m,))
    for k in range(1,n+1):
        # v = np.dot(A,Q[:,k-1])  # Generate a new candidate vector
        v = funA(Q[:,k-1].reshape(mm)).reshape((m,))

        for j in range(k):  # Subtract the projections on previous vectors
            h[j,k-1] = np.sum(mul(Q[:,j],v))
            v = v - mul(h[j,k-1],Q[:,j])
        h[k,k-1] = np.linalg.norm(v,2)
        if h[k,k-1] > eps:  # Add the produced vector to the list, unless
            Q[:,k] = v/h[k,k-1]
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h
