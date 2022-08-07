import numpy as np
def Lanczos(A,b,N):
    c=b/np.linalg.norm(b)
    Q=np.zeros((N+1,len(b)))
    Q[0,:]=c
    alpha=np.zeros([N,])
    beta=np.zeros([N,])
    for j in range(N):
        w=A@Q[j,:]
        # orthogonalize
        alpha[j]=np.dot(Q[j,:].T,w)
        if j==0:
            qtilde=w-alpha[j]*Q[j,:]
        else:
            qtilde=w-alpha[j]*Q[j,:]-beta[j-1]*Q[j-1,:]
        # normalize
        beta[j]=np.linalg.norm(qtilde)
        q=qtilde/beta[j]
        Q[j+1,:]=q
    return alpha,beta

rng = np.random.default_rng(0)
A = rng.standard_normal((10,10))
A = A + A.T
q = rng.standard_normal(10,)
alpha,beta = Lanczos(A,q,10)
T=np.diag(alpha)+np.diag(beta[:-1],-1)+np.diag(beta[:-1],1)
approx_eigs = np.linalg.eig(T)[0]
print('approx_eigs',approx_eigs)
eigs = np.linalg.eig(A)[0]
print('eigs',eigs)