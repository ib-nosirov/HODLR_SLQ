import numpy as np
from numpy import linalg as LA
import scipy.io
import HODLR_FMM
import seaborn as sns; sns.set(context="talk",style="white",palette="pastel")
import matplotlib.pyplot as plt
# from here https://scicomp.stackexchange.com/questions/23536/quality-of-eigenvalue-approximation-in-lanczos-method
def HODLRLanczos(A,v,m=100):
	n = len(v)
	if m>n:
		m = n
	# from here https://en.wikipedia.org/wiki/Lanczos_algorithm
	V = np.zeros((m,n))
	T = np.zeros((m,m))
	v0 = np.zeros(n)
	beta = 0
	for j in range(m-1):
		w = HODLR_FMM.HODLR_matvec(A,v)
		alpha = np.dot(w,v)
		print(alpha)
		w = w - alpha*v - beta*v0
		beta = np.sqrt(np.dot(w,w)) 
		v0 = v
		v = w/beta 
		T[j,j] = alpha 
		T[j,j+1] = beta
		T[j+1,j] = beta
		V[j,:] = v
	w = HODLR_FMM.HODLR_matvec(A,v)
	alpha = np.dot(w,v)
	w = w - alpha*v - beta*v0
	T[m-1,m-1] = np.dot(w,v)
	V[m-1] = w / np.sqrt(np.dot(w,w)) 
	return T,V

def Lanczos(A,v,m=100):
	n = len(v)
	if m>n:
		m = n
	# from here https://en.wikipedia.org/wiki/Lanczos_algorithm
	V = np.zeros((m,n))
	T = np.zeros((m,m))
	v0 = np.zeros(n)
	beta = 0
	for j in range(m-1):
		w = np.dot(A,v)
		alpha = np.dot(w,v)
		w = w - alpha*v - beta*v0
		beta = np.sqrt(np.dot(w,w)) 
		v0 = v
		v = w/beta 
		T[j,j] = alpha 
		T[j,j+1] = beta
		T[j+1,j] = beta
		V[j,:] = v
	w = np.dot(A,v)
	alpha = np.dot(w,v)
	w = w - alpha*v - beta*v0
	T[m-1,m-1] = np.dot(w,v)
	V[m-1] = w / np.sqrt(np.dot(w,w)) 
	return T,V

def makeA():
	N = 1000
	A = np.zeros((N,N))
	sigma = N/10
	for ii in range(N):
		for jj in range(N):
			A[ii,jj] = np.exp(-(ii-jj)**2/sigma**2)
	return A

HODLR_mtrx = scipy.io.loadmat('HODLR_mtrx.mat')
#y_exact = scipy.io.loadmat('y.mat')
u_tree = HODLR_mtrx['u_tree']
z_tree = HODLR_mtrx['z_tree']
leaves_cell = HODLR_mtrx['leaves_cell']
idx_tree = HODLR_mtrx['idx_tree']
b = np.ones(2*u_tree[0,1].shape[0])
A = [u_tree,z_tree,leaves_cell,idx_tree]
#HODLRT,HODLRV = HODLRLanczos(A,b)
A = makeA()
T,V = Lanczos(A,b)
#ax = sns.heatmap(T,cmap="Pastel1",cbar=False,)
#ax = ax.set(title='Tridiagonal',xticklabels=[],yticklabels=[])
#plt.show()
#ax = sns.heatmap(V,cmap="Pastel1",cbar=False,)
#ax = ax.set(title='Mutually orthogonal vectors',xticklabels=[],yticklabels=[])
#plt.show()
ax = sns.heatmap(np.dot(V,np.dot(A,V.transpose())),cmap="Pastel1",cbar=False,)
ax = ax.set(title='V*AV',xticklabels=[],yticklabels=[])
plt.show()
#print(y)
#print(LA.norm(y_exact['y'].T-y,'fro')/LA.norm(y_exact['y'],'fro'))
