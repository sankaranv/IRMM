import numpy as np

def computeStochasticMatrix_hypergraph(H, W):

	# Vertex degree vector
	dv = np.sum(np.dot(H, np.diag(W)), axis=1, keepdims=True)
	invDv = np.diag(np.linalg.inv(dv))
	Pi = dv/np.sum(dv)
	De = np.diag(np.sum(H, axis=1, keepdims=True) - 1)
	invDe = np.linalg.inv(De)
	theta = np.dot(np.dot(np.dot(invDv,H),invDe),H.T)
	# Set diagonal elements to zero
	theta = np.fill_diagonal(theta,0)
	return theta, Pi

def computeNormalizedLaplacian_hypergraph(H, W):

	n = H.shape[0]
	dv = np.sum(H,axis=1, keepdims=True) # Vertex degree vector
	invdv = 1/np.sqrt(dv)
	invDv = np.diag(invdv)
	Dv = np.diag(dv)
	de = np.sum(H,axis=0, keepdims=True) - 1 # Edge degree vector
	invde = 1/de
	invDe = np.diag(invde)
	# invDv * H * W * invDe * H.T * invDv
	L1 = np.dot(invDv,np.dot(H,np.dot(np.diag(W),np.dot(invDe,np.dot(H.T,invDv)))))
	L = np.eye(n) - L1
	return L, L1, Dv, invDv, invDe

def computeScholkopfLaplacian_hypergraph(H, W):

	n = H.shape[0]
	dv = np.sum(H,axis=1, keepdims=True) # Vertex degree vector
	invdv = 1/np.sqrt(dv)
	invDv = np.diag(invdv)
	Dv = np.diag(dv)
	de = np.sum(H,axis=0, keepdims=True) # Edge degree vector
	invde = 1/de
	invDe = np.diag(invde)
	# invDv * H * W * invDe * H.T * invDv;
	L1 = np.dot(invDv,np.dot(H,np.dot(np.diag(W),np.dot(invDe,np.dot(H.T,invDv)))))
	L = np.eye(n) - L1
	mask = np.eye(L.shape[0]).astype(bool) # Select diagonal elements
	L[mask] = 0 
	return L, L1, Dv, invDv, invDe

def computeHybridLaplacian_hypergraph(H, W, weighted=1):

	# If weighted = 1, use weighted clique formulation, else normal
	n = H.shape[0]
	De = np.sum(H, size=0, keepdims=True)
	De = De.T
	W = W * (1/(De-1))
	norm_W = W
	dv = np.sum(H, axis=1, keepdims=True)
	invdv = 1/dv
	invDv = np.diag(invdv)
	de = np.sum(H, axis=0, keepdims=True) - 1
	invde = 1/de
	invDe = np.diag(invde)
	Dv = np.diag(dv)
	A = np.dot(H,np.dot(np.diag(W),H.T))
	# A = A - Dv (This is used in Harini's thesis but not coded)
	mask = np.eye(A.shape[0]).astype(bool)
	A[mask] = 0
	L = np.eye(n) - np.dot(invDv,A)
	return L, Dv, invDv, invDe

def computeNormalizedAdjacency_hypergraph(H, W):

	n = H.shape[0]
	dv = np.sum(H, axis=1, keepdims=True)
	invdv = 1/np.sqrt(dv)
	invDv = np.diag(invdv)
	Dv = np.diag(dv)
	de = np.sum(H, axis=0, keepdims=True) - 1
	invde = 1/de
	invDe = np.diag(invde)
	L1 = np.dot(H, np.dot(np.diag(W), np.dot(invDe, H.T)))
	mask = np.eye(L1.shape[0]).astype(bool)
	L1[mask] = 0
	L = np.eye(n) - L1
	return L, L1, Dv, invDv, invDe

def computeIncrementalLaplacian_hypergraph(H, W, invDv, invDe):

	# L1 = invDv * H * W * invDe * H.T * invDv
	L1 = np.dot(invDv,np.dot(H,np.dot(np.diag(W),np.dot(invDe,np.dot(H.T,nvDv)))))
	return L

def computeIncrementalLaplacian_hypergraph_louvain(H, W, invDv, invDe, weighted):

	n = H.shape[0]
	De = np.sum(H, axis=0, keepdims=True)
	De = De.T
	if (weighted != 0):
		W = W * (1/(De - 1))
	norm_W = W
	dv = np.sum(H, axis=1, keepdims=True)
	invdv = 1/np.sqrt(dv)
	invDv = np.diag(invdv)	
	de = np.sum(H, axis=0, keepdims=True) - 1
	invde = 1/de
	invDe = np.diag(invde)
	Dv = np.diag(dv)
	A = np.dot(H,np.dot(np.diag(W),H.T))
	mask = np.eye(A.shape[0]).astype(bool)
	A[mask] = 0
	L2 = np.eye(n) - np.dot(invDv,A)
	return L2

def computeIncremental_Adjacency_hypergraph(H, W, invDv, invDe):

	dv = np.sum(H, axis=1, keepdims=True)
	L1 = np.dot(H,np.dot(np.diag(W),np.dot(invDe,H.T))) - np.diag(dv)
	mask = np.eye(L1.shape[0]).astype(bool)
	L1[mask] = 0
	return L1

def computeAdjacencyMatrix_hypergraph(H, W, weighted):

	n = H.shape[0]
	De = np.sum(H, axis=0, keepdims=True)
	De = De.T
	if (weighted != 0):
		W = W * (1/(De - 1))
	norm_W = W
	dv = np.sum(H, axis=1, keepdims=True)
	invdv = 1/np.sqrt(dv)
	invDv = np.diag(invdv)
	de = np.sum(H, axis=0, keepdims=True) - 1
	invde = 1/de
	invDe = np.diag(invde)
	Dv = diag(dv)
	A = np.dot(H,np.dot(np.diag(W), np.dot(invDe, H.T)))
	mask = np.eye(A.shape[0]).astype(bool)
	A[mask] = 0
	return A, invDv, invDe, Dv, norm_W

def computeAdjacencyMatrix_for_modularity_hypergraph(H, W, weighted):

	n = H.shape[0]
	De = np.sum(H, axis=0, keepdims=True)
	De = De.T
	W = W * (1/(De - 1))
	norm_W = W
	dv = np.sum(H, axis=1, keepdims=True)
	invdv = 1/np.sqrt(dv)
	invDv = np.diag(invdv)
	de = np.sum(H, axis=0, keepdims=True) - 1
	invde = 1/de
	invDe = np.diag(invde)
	Dv = diag(dv)
	A = np.dot(H,np.dot(np.diag(W), H.T))
	mask = np.eye(A.shape[0]).astype(bool)
	A[mask] = 0
	return A, invDv, invDe, Dv, norm_W			

def computeCliqueMatrix_for_modularity_hypergraph(H, W, weighted):

	n = H.shape[0]
	De = np.sum(H, axis=0, keepdims=True)
	De= De.T
	if (weighted != 0):
		W = W * (1/(De - 1))
	norm_W = W
	A = np.dot(H,H.T)
	dv = np.sum(H, axis=1, keepdims=True)
	invdv = 1/np.sqrt(dv)
	invDv = np.diag(invdv)
	invde = 1/(De-1)
	invDe = np.diag(invde)
	Dv = diag(dv)
	mask = np.eye(A.shape[0]).astype(bool)
	A[mask] = 0
	return A, invDv, invDe, Dv, norm_W

def computeCliqueMatrix_for_laplacian_hypergraph(H, W, weighted):

	n = H.shape[0]
	De = np.sum(H, axis=0, keepdims=True)
	De= De.T
	if (weighted != 0):
		W = W * (1/(De - 1))
	norm_W = W
	A = np.dot(H,H.T)
	dv = np.sum(H, axis=1, keepdims=True)
	invdv = 1/np.sqrt(dv)
	invDv = np.diag(invdv)
	invde = 1/(De-1)
	invDe = np.diag(invde)
	Dv = diag(dv)
	mask = np.eye(A.shape[0]).astype(bool)
	A[mask] = 0
	L = np.eye(n) - np.dot(invDv,A)	
	return L, invDv, invDe, Dv, norm_W

def computeModularityMatrix_hypergraph(H, W, weighted):

	A = computeAdjacencyMatrix_hypergraph(H, W, weighted)
	M = computeModularityforAdjacency(A)
	return M

def computeModularityforAdjacency(A):

	k = np.sum(A, axis=1, keepdims=True)
	m = np.sum(k)
	P = np.dot(k,k.T)/m
	M = A - P
	return M

def assignGroups(Y):

	pass
	# return S, groups, tempGroups


def convertAdjacencyToIncidence(A):

	n = A.shape[0]
	H = np.empty(shape=[n, 0])
	e = 0
	for i in range (n):
		for j in range(i+1,n):
			if (A[i][j]==1):
				# H[:,e] = np.zeros((n, 1))
				H = np.append(H, np.zeros((n, 1)), axis=1)
				H[i,e] = 1
				H[j,e] = 1
				e = e + 1
	return H

def updateWeights(H, W, clusters):

	m = H.shape[1]
	n = H.shape[0]
	k = np.amax(clusters)
	changeInWeights = np.zeros((m, 1))
	for i in range(m):
		nodesInHyperedge = np.where(H[:,i] == 1)[0]	
		nodeClusters = clusters[nodesInHyperedge]
		numNodesInHyperedge = len(nodesInHyperedge)
		for j in range(k):
			numNodesInCluster = np.sum(nodeClusters == j)
			changeInWeights[i] = changeInWeights[i] + (1 / (numNodesInCluster + 1))
		changeInWeights[i] = changeInWeights[i] / (1 / (numNodesInHyperedge + k))
	changeInWeights = m * changeInWeights / np.sum(changeInWeights)
	newW = (changeInWeights + W) / 2
	return newW

















