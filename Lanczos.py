import numpy as np
from numpy import linalg as LA
import scipy.io
import HODLR_FMM
import seaborn as sns; sns.set(context="talk",style="white",palette="pastel")
import matplotlib.pyplot as plt

def HODLRLanczos(A,b,N):
    c = b / np.linalg.norm(b)
    Q = np.zeros((N+1,len(b)))
    Q[0,:] = c
    alpha = np.zeros([N,])
    beta = np.zeros([N,])
    for j in range(N):
        w = HODLR_FMM.HODLR_matvec(A,Q[j,:])
        # orthogonalize
        alpha[j] = np.dot(Q[j,:].T,w)
        if j==0:
            qtilde = w - alpha[j]*Q[j,:]
        else:
            qtilde = w - alpha[j]*Q[j,:] - beta[j-1]*Q[j-1,:]
        # normalize
        beta[j]=np.linalg.norm(qtilde)
        q = qtilde / beta[j]
        Q[j+1,:] = q
    return alpha,beta

def Lanczos(A,b,N):
    c = b/np.linalg.norm(b)
    Q = np.zeros((N+1,len(b)))
    Q[0,:] = c
    alpha=np.zeros([N,])
    beta=np.zeros([N,])
    for j in range(N):
        w = A@Q[j,:]
        # orthogonalize
        alpha[j] = np.dot(Q[j,:].T,w)
        if j==0:
            qtilde = w - alpha[j]*Q[j,:]
        else:
            qtilde = w - alpha[j]*Q[j,:] - beta[j-1]*Q[j-1,:]
        # normalize
        beta[j] = np.linalg.norm(qtilde)
        q = qtilde / beta[j]
        Q[j+1,:] = q
    return alpha,beta
	
n = 50
rng = np.random.default_rng(0)
A = rng.standard_normal((n,n))
A = A + A.T
q = rng.standard_normal(n,)
alpha,beta = Lanczos(A,q,n)
T=np.diag(alpha)+np.diag(beta[:-1],-1)+np.diag(beta[:-1],1)
#approx_eigs = np.linalg.eig(T)[0]
#print('approx_eigs',np.sort(approx_eigs))
#eigs = np.real_if_close(np.linalg.eig(A)[0])
#print('eigs',np.sort(eigs))
#print(np.sort(approx_eigs)-np.sort(eigs))

def makeA():
	N = 1000
	A = np.zeros((N,N))
	sigma = N/10
	for ii in range(N):
		for jj in range(N):
			A[ii,jj] = np.exp(-(ii-jj)**2/sigma**2)
	return A

HODLR_mtrx = scipy.io.loadmat('HODLR_mtrx.mat')
u_tree = HODLR_mtrx['u_tree']
z_tree = HODLR_mtrx['z_tree']
leaves_cell = HODLR_mtrx['leaves_cell']
idx_tree = HODLR_mtrx['idx_tree']
b = np.ones(2*u_tree[0,1].shape[0])
A = [u_tree,z_tree,leaves_cell,idx_tree]
HODLR_eye = HODLR_FMM.HODLR_matvec(A,np.eye(1000)[0,:])
alpha,beta = HODLRLanczos(A,b,10) 
T=np.diag(alpha)+np.diag(beta[:-1],-1)+np.diag(beta[:-1],1)
approx_eigs = np.linalg.eig(T)[0]
A = makeA()
eigs = np.real_if_close(scipy.sparse.linalg.eigs(A)[0])
#print('eigs',eigs)
approx_eigs = np.sort(approx_eigs)
eigs = np.sort(eigs)
print('approx_eigs',approx_eigs[:10])
print('eigs',eigs)
#print(approx_eigs[:10]-eigs[:10])