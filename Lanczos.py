import numpy as np
from numpy import linalg as LA
import scipy.io
from scipy.sparse.linalg import LinearOperator
import HODLR_FMM
# from here https://scicomp.stackexchange.com/questions/23536/quality-of-eigenvalue-approximation-in-lanczos-method
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
		w = HODLR_FMM.HODLR_matvec(A,v)
		alpha = np.dot(w,v)
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

HODLR_mtrx = scipy.io.loadmat('HODLR_mtrx.mat')
#y_exact = scipy.io.loadmat('y.mat')
u_tree = HODLR_mtrx['u_tree']
z_tree = HODLR_mtrx['z_tree']
leaves_cell = HODLR_mtrx['leaves_cell']
idx_tree = HODLR_mtrx['idx_tree']
b = np.ones(2*u_tree[0,1].shape[0])
A = [u_tree,z_tree,leaves_cell,idx_tree]
T,V = Lanczos(A,b)
print(T)
print(V)
#print(y)
#print(LA.norm(y_exact['y'].T-y,'fro')/LA.norm(y_exact['y'],'fro'))
