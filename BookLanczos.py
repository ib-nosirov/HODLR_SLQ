import numpy as np
from numpy import linalg as LA

def BookLanczos(A,q1):
    n = len(q1)
    # Collect the alphas and betas in this array for now.
    # We don't know how many there will be because of the while loop.
    T = np.zeros(A.shape)
    w = q1
    v = A@w
    alpha = np.dot(w,v)
    v -= alpha*w
    beta = LA.norm(v)
    k = 0
    T[k,k] = alpha
    T[k,k+1] = beta
    T[k+1,k] = beta
    while abs(beta) > 0.1:
        k += 1
        for i in range(n):
            t = w[i]
            w[i] = v[i]/beta
            v[i] = -beta*t
        v += A@w
        alpha = np.dot(w,v)
        print('alpha: ',alpha)
        v -= alpha*w
        print('v: ',v)
        beta = LA.norm(v)
        print('beta: ',beta)
        T[k,k] = alpha
        T[k,k+1] = beta
        T[k+1,k] = beta
    T = T[:k+1,:k+1]
    return T

#rng = np.random.default_rng(0)
A = np.ones((10,10))
A = A + A.T
q = np.ones(10,)
T = BookLanczos(A,q)
print(T)
approx_eigs = LA.eig(T)[0]
print('approx_eigs',approx_eigs)
eigs = LA.eig(A)[0]
print('eigs',eigs)