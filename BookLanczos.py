import numpy as np
from numpy import linalg as LA

def BookLanczos(A,q1):
    n = len(q1)
    T = np.zeros((2,n))
    w = q1
    v = A@w
    alpha = np.dot(w,v)
    v -= alpha*w
    beta = LA.norm(v)
    k = 1
    while abs(beta) > 0.1:
        for i in range(n):
            t = w[i]
            w[i] = v[i]/beta
            v[i] = -beta*t
        v += A@w
        k += 1
        alpha = np.dot(w,v)
        v -= alpha*w
        beta = LA.norm(v)
        T[0,k-2] = alpha
        T[1,k-2] = beta
    return T

rng = np.random.default_rng(0)
A = rng.standard_normal((10,10))
A = A + A.T
q = rng.standard_normal(10,)
T = BookLanczos(A,q)
print(T)
#approx_eigs = LA.eig(T)[0]
#print('approx_eigs',approx_eigs)
#eigs = LA.eig(A)[0]
#print('eigs',eigs)