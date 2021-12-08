import numpy as np

def gauss_lobatto(p):
    '''
    Compute the p+1 Gauss-Lobatto-Legendre nodes z on [-1,1],
    i.e. the zeros of the first derivative of the Legendre
    polynomial of degree p plus -1 and 1 and the p+1 weights w.
    Coded by Alexey Voronin: https://github.com/lexeyV/
    '''
    n = p + 1
    z = np.zeros((n,))
    w = np.zeros((n,))
    z[0] = -1 
    z[-1] = 1
    if p > 1: 
        if p == 2: 
            z[1] = 0
        else:
            M = np.zeros((p - 1, p - 1))
            for i in range(1, p - 2 + 1):  # check the range
                M[i - 1, i] = (1.0 / 2.0) * np.sqrt(
                    (i * (i + 2.0)) / ((i + 1.0 / 2.0) * (i + 3.0 / 2.0))
                )
                M[i, i - 1] = M[i - 1, i]
            eigvals, eigvecs = np.linalg.eig(M)
            z[1:p] = np.sort(eigvals)
    # compute the weights w
    w[0] = 2.0 / (p * (n))
    w[n - 1] = w[0]
    for i in range(1, p):  # check range
        x = z[i]
        z0 = 1
        z1 = x
        for j in range(0, p - 1):
            z2 = x * z1 * (2.0 * (j + 1) + 1.0) / ((j + 1) + 1.0) - z0 * (j + 1) / (
                (j + 1) + 1.0
            )
            z0 = z1
            z1 = z2
        w[i] = 2.0 / (p * (n) * z2 * z2)
    return z, w

    
def lagrange_derivative_matrix(x):
    assert x.ndim == 1

    n = x.size

    a = np.ones((n,), dtype=np.float64)

    for i in range(n):
        for j in range(i):
            a[i] = a[i] * (x[i] - x[j])
        for j in range(i + 1, n):
            a[i] = a[i] * (x[i] - x[j])
    a = 1.0 / a
    D = np.zeros((n, n))

    for i in range(n):
        D[i, :] = a[i] * (x[i] - x)
        D[i, i] = 1
    for j in range(n):
        D[:, j] = D[:, j] / a[j]
    D = 1.0 / D

    for i in range(n):
        D[i, i] = 0
        D[i, i] = -sum(D[i, :])
    return D


def reference_derivative_matrix(p):
    z, w = gauss_lobatto(p)
    D = lagrange_derivative_matrix(z)
    return D

def semhat(N):
    z, w = gauss_lobatto(N)
    D = lagrange_derivative_matrix(z)
    return z, w, D



