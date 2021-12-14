import numpy as np

'''
Build 1D interpolation matrix from two arbitary point sets
Re-write matlab code into python
'''

def interp_mat(xo,xi):
    '''
    Compute the interpolation matrix from xi to xo
    '''
      
    no = len(xo)
    ni = len(xi)
    Ih = np.zeros((ni,no))
    w  = np.zeros((ni,2))

    for i in range(no):
       w = fd_weights_full(xo[i],xi,1)
       Ih[:,i] = w[:,0]

    Ih = Ih.T

    return Ih


def fd_weights_full(xx,x,m):
    '''
    This routine evaluates the derivative based on all points
    in the stencils.  It is more memory efficient than "fd_weights"

    This set of routines comes from the appendix of 
    A Practical Guide to Pseudospectral Methods, B. Fornberg
    Cambridge Univ. Press, 1996.   (pff)

    Input parameters:
      xx -- point at wich the approximations are to be accurate
      x  -- array of x-ordinates:   x(0:n)
      m  -- highest order of derivative to be approxxmated at xi

    Output:
      c  -- set of coefficients c(0:n,0:m).
            c(j,k) is to be applied at x(j) when
            the kth derivative is approximated by a 
            stencil extending over x(0),x(1),...x(n).


    UPDATED 8/26/03 to account for matlab "+1" index shift.
    Follows p. 168--169 of Fornberg's book.

    UPDATED 12/09/21 implement in python, back to 0 based index (ylan)
    '''

    n1 = len(x)
    m1 = m+1

    c1       = 1.0
    c4       = x[0] - xx

    c = np.zeros((n1,m1))
    c[0,0] = 1.0

    for i in range(1,n1,1):
       mn = min(i,m)
       c2 = 1.0
       c5 = c4
       c4 = x[i]-xx
       for j in range(i):
          c3 = x[i]-x[j]
          c2 = c2*c3
          for k in range(mn,0,-1):
             c[i,k] = c1*(k*c[i-1,k-1]-c5*c[i-1,k])/c2;
          c[i,0] = -c1*c5*c[i-1,0]/c2;
          for k in range(mn,0,-1):
             c[j,k] = (c4*c[j,k]-k*c[j,k-1])/c3
          c[j,0] = c4*c[j,0]/c3
       c1 = c2

    return c


def test_interp(Nf=80,Nc=15):
    import matplotlib.pyplot as plt
    from my_sem.gen_semhat import semhat

    zf, wf, Df = semhat(Nf)
    zc, wc, Dc = semhat(Nc)

    uf = np.sin(3 * np.pi * zf)
    uc = np.sin(3 * np.pi * zc)
      
    Jh = interp_mat(zf,zc)

    u_c2f = Jh @ uc

    plt.figure()
    plt.plot(zc,uc, 'o', color='tab:orange', clip_on=False, ms=15,label='coarse')
    plt.plot(zf,u_c2f, 'o-', clip_on=False,label='fine (from coarse)')
    plt.plot(zf,uf, '-', color='red', clip_on=False, label='fine (exact)')
    err = np.max(np.abs(u_c2f - uf))
    plt.legend();plt.title(r'($N_c$, $N_f$)=(%d,%d)  err= %2.2e'%(Nc,Nf,err))

    Jh_i = interp_mat(zc,zf)

    u_f2c = Jh_i @ uf

    plt.figure()
    plt.plot(zf,uf, 'o', color='tab:orange', clip_on=False, ms=15,label='fine')
    plt.plot(zc,u_f2c, 'o-', clip_on=False,label='coarse (from fine)')
    plt.plot(zc,uc, '-', color='red', clip_on=False, label='coarse (exact)')
    err = np.max(np.abs(u_f2c - uc))
    plt.legend();plt.title(r'($N_c$, $N_f$)=(%d,%d)  err= %2.2e'%(Nc,Nf,err))

    plt.show()
    quit()




