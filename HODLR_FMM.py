import numpy as np
from numpy import linalg as LA
import scipy.io
def HODLR_matvec(A,b):
	# ARGS:
	#	u_tree: 3d array containing u matrices.
	#	z_tree: 3d array containing z matrices.
	#	A_diags: main block-diagonal entries. 
	#	it_arr: breadth-first iterator array containing nodes.
	# *Note: the first node in each tree has been removed while processing.
	u_tree = A[0]
	z_tree = A[1]
	leaves_cell = A[2]
	idx_tree = A[3]

	# determine the number of nodes.
	num_nodes = idx_tree.shape[1]
	# create a placeholder nxk vector; call it 'y'.
	output_length = 2*u_tree[0,1].shape[0]
	y = np.zeros(output_length)
	# for each layer: compute u_i (z_i* b_i) where * is the conjugate
	# transpose.
	layers_arr = get_layers(num_nodes)
	offset = 0
	for i_layer in range(len(layers_arr)-1):
		# We will populate this matrix by stacking matrix multiplies.
		for j_node in get_nodes(i_layer,layers_arr,num_nodes):
			offset = j_node
			start = idx_tree[0,j_node][0,0]-1
			finish = idx_tree[0,j_node][0,1]
			if j_node%2 == 0:
				u = u_tree[0,j_node]
				z = z_tree[0,j_node+1]
			else:
				u = u_tree[0,j_node]
				z = z_tree[0,j_node-1]
		# Stack the computed vectors at the current layer into one nxk vector.
			z_b = z.T.dot(b[start:finish])
			y[start:finish] += u.dot(z_b)
	# add the leaves	
	for i_node in range(len(leaves_cell[0])):
		start = idx_tree[0,i_node+offset][0,0]-1
		finish = idx_tree[0,i_node+offset][0,1]
		diag_block = leaves_cell[0,i_node]
		y[start:finish] += diag_block.dot(b[start:finish])
	return y

def get_layers(num_nodes):
	# leverage the fact that this is a binary tree and compute where each layer
	# begins.
	num_layers = int(np.log2(num_nodes))
	layers_arr = np.zeros(num_layers,'int16')
	for node_idx in range(2,num_nodes+1):
		arr_idx = np.log2(node_idx)
		if arr_idx%1==0:
			layers_arr[int(arr_idx)-1] = node_idx-2
	return layers_arr

def get_nodes(layer,layers_arr,num_nodes):
	nodes_arr = np.empty(0)
	if layer+1 < layers_arr.size:
		nodes_arr = np.arange(layers_arr[layer],layers_arr[layer+1])
	else:
		nodes_arr = np.arange(layers_arr[layer],num_nodes)
	return nodes_arr
 
