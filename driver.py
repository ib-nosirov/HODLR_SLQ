import numpy as np
import HODLR_FMM
from numpy import linalg as LA
import scipy.io
HODLR_mtrx = scipy.io.loadmat('HODLR_mtrx.mat')
y_exact = scipy.io.loadmat('y.mat')
u_tree = HODLR_mtrx['u_tree']
z_tree = HODLR_mtrx['z_tree']
leaves_cell = HODLR_mtrx['leaves_cell']
idx_tree = HODLR_mtrx['idx_tree']
b = np.ones(2*u_tree[0,1].shape[0])
A = [u_tree,z_tree,leaves_cell,idx_tree]
y = HODLR_FMM.HODLR_matvec(A,b)
print(y_exact['y'].T-y)