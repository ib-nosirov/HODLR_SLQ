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

def Lanczos(A,q,m=100):
# pass-complexity: w = A@q and is sequential
# with HODLR, 
	n = len(q)
	if m>n:
		m = n
	# from here https://en.wikipedia.org/wiki/Lanczos_algorithm
	V = np.zeros((n,m))
	T = np.zeros((m,m))
	q0 = np.zeros(n)
	beta = 0
	for j in range(m-1):
		w = A@q
		alpha = w@q 
# w -= alpha*q + beta*q0 may have different effect because += and -= overwrites
# memory. 
		w = w - alpha*q - beta*q0
		beta = np.linalg.norm(w)
		q0 = q
		q = w/beta 
		T[j,j] = alpha 
		T[j+1,j] = beta
		T[j,j+1] = beta
		V[:,j] = q
#		print(V.T@A@V - T)
	w = A@q
	alpha = w@q
	w = w - alpha*q - beta*q0
	T[m-1,m-1] = w@q
	V[:,m-1] = w / np.linalg.norm(w)
#	print(V)
#	print(V.T@A@V - T)
	return T,V
# 2d array is different than (10,) different than (10,1)

rng = np.random.default_rng(0)
A = rng.standard_normal((10,10))
A = A + A.T
q = rng.standard_normal(10,)
T,V = Lanczos(A,q,m=10)
approx_eigs = LA.eig(T)[0]
print('approx_eigs',approx_eigs)
eigs = LA.eig(A)[0]
print('eigs',eigs)