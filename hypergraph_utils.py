import numpy as np

def computeStochasticMatrix(H, W):

	# Vertex degree vector
	dv = np.sum(np.dot(H, np.diag(W)), axis=1)
	invDv = np.diag(np.linalg.inv(dv))
	Pi = dv/np.sum(dv)
	De = np.diag(np.sum(H, axis=1) - 1)
	invDe = np.linalg.inv(De)
	theta = np.dot(np.dot(np.dot(invDv,H),invDe),H.T)
	# Set diagonal elements to zero
	theta = np.fill_diagonal(theta,0)
	return theta, Pi

def computeNormalizedLaplacian(H, W):

	n = H.shape[0]
	dv = np.sum(H,axis=1) # Vertex degree vector
	invdv = 1/np.sqrt(dv)
	invDv = np.diag(invdv)
	Dv = np.diag(dv)
	de = np.sum(H,axis=0) - 1 # Edge degree vector
	invde = 1/de
	invDe = np.diag(invde)
	# invDv * H * W * invDe * H.T * invDv
	L1 = np.dot(invDv,np.dot(H,np.dot(np.diag(W),np.dot(invDe,np.dot(H.T,invDv)))))
	L = np.eye(n) - L1
	return L, L1, Dv, invDv, invDe

def computeScholkopfLaplacian(H, W):

	n = H.shape[0]
	dv = np.sum(H,axis=1) # Vertex degree vector
	invdv = 1/np.sqrt(dv)
	invDv = np.diag(invdv)
	Dv = np.diag(dv)
	de = np.sum(H,axis=0) # Edge degree vector
	invde = 1/de
	invDe = np.diag(invde)
	# invDv * H * W * invDe * H.T * invDv;
	L1 = np.dot(invDv,np.dot(H,np.dot(np.diag(W),np.dot(invDe,np.dot(H.T,invDv)))))
	L = np.eye(n) - L1
	mask = np.eye(L.shape[0]).astype(bool) # Select diagonal elements
	L[mask] = 0 
	return L, L1, Dv, invDv, invDe

def computeHybridLaplacian(H, W, weighted=1):

	# If weighted = 1, use weighted clique formulation, else normal
	n = H.shape[0]
	De = np.sum(H, size=0)
	De = De.T
	W = W * (1/(De-1))
	norm_W = W
	dv = np.sum(H, axis=1)
	invdv = 1/dv
	invDv = np.diag(invdv)
	de = np.sum(H, axis=0) - 1
	invde = 1/de
	invDe = np.diag(invde)
	Dv = np.diag(dv)
	A = np.dot(H,np.dot(np.diag(W),H.T))
	# A = A - Dv (This is used in Harini's thesis but not coded)
	mask = np.eye(A.shape[0]).astype(bool)
	A[mask] = 0
	L = np.eye(n) - np.dot(invDv,A)
	return L, Dv, invDv, invDe

def computeNormalizedAdjacency(H, W):

	n = H.shape[0]
	dv = np.sum(H, axis=1)
	invdv = 1/np.sqrt(dv)
	invDv = np.diag(invdv)
	Dv = np.diag(dv)
	de = np.sum(H, axis=0) - 1
	invde = 1/de
	invDe = np.diag(invde)
	L1 = np.dot(H, np.dot(np.diag(W), np.dot(invDe, H.T)))
	mask = np.eye(L1.shape[0]).astype(bool)
	L1[mask] = 0
	L = np.eye(n) - L1
	return L, L1, Dv, invDv, invDe

def computeIncrementalLaplacian(H, W, invDv, invDe):

	# L1 = invDv * H * W * invDe * H.T * invDv
	L1 = np.dot(invDv,np.dot(H,np.dot(np.diag(W),np.dot(invDe,np.dot(H.T,nvDv)))))
	return L


