import numpy as np
from numpy import linalg as LA
from scipy.linalg import schur, eigvals
def Lanczos(A,w,m):
    n = len(w)
    if m>n:
        m = n
    v = A@w
    T = np.zeros((m,m))
    V = np.zeros((m,m))
    q0 = np.zeros(n)
    beta = LA.norm(v)
    for i in range(m-1):
        tmp = w
        w = v/beta
        v = -beta*tmp
        v += A@w 
        alpha = np.dot(w.T,v)
        v -= alpha*w
        beta = LA.norm(v)
        T[i,i] = alpha
        T[i+1,i] = beta
        T[i,i+1] = beta
        V[i] = v
    t = w
    w = v/beta
    v = -beta*t
    v += A@w 
    alpha = np.dot(w.T,v)
    v -= alpha*w
    V[-1] = v
    T[m-1,m-1] = alpha
    print('isV_Orthonormal', V@V.T)
    return T

rng = np.random.default_rng(0)
A = rng.standard_normal((10,10))
A = A + A.T
q = rng.standard_normal(10,)
n = 10
T = Lanczos(A,q,n)
approx_eigs = LA.eig(T)[0]
print('approx_eigs',approx_eigs)
eigs = LA.eig(A)[0]
print('eigs',eigs)