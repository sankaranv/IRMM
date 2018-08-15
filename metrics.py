def measureRandIndex(clusterLabels, groundTruth):

	n = clusterLabels.shape[0]
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for i in range(n):
		for j in range(i+1,n):
			if (clusterLabels[i] == clusterLabels[j]):
				if (groundTruth[i] == groundTruth[j]):
					TP = TP + 1
				else:
					FP = FP + 1
			else:
				if (groundTruth[i] == groundTruth[j]):
					FN = FN + 1
				else:
					TN = TN + 1
	RI = 2*(TP + TN)/(n*(n - 1))
	return RI

def measureF1(clusterLabels, groundTruth):

	# Bidirectional F1 score
	n = clusterLabels.shape[0]
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for i in range(n):
		for j in range(i+1,n):
			if (clusterLabels[i] == clusterLabels[j]):
				if (groundTruth[i] == groundTruth[j]):
					TP = TP + 1
				else:
					FP = FP + 1
			else:
				if (groundTruth[i] == groundTruth[j]):
					FN = FN + 1
				else:
					TN = TN + 1
	Precision_1 = TP/(TP + FP)
	Recall_1 = TP/(TP + FN)	
	F1_a = 2 * (Precision_1 * Recall_1) / (Precision_1 + Recall_1)

	for i in range(n):
		for j in range(i+1,n):
			if (groundTruth[i] == groundTruth[j]):
				if (clusterLabels[i] == clusterLabels[j]):
					TP = TP + 1
				else:
					FP = FP + 1
			else:
				if (clusterLabels[i] == clusterLabels[j]):
					FN = FN + 1
				else:
					TN = TN + 1
	Precision_2 = TP/(TP + FP)
	Recall_2 = TP/(TP + FN)	
	F1_b = 2 * (Precision_2 * Recall_2) / (Precision_2 + Recall_2)
	F1 = (F1_a+F1_b)/2
	return F1

def calculatePurity(clusterAssignments, groundTruth):

	k = np.amax(clusterAssignments)
	c = np.amax(groundTruth)
	assign = np.zeros((k, 1))
	purity = np.zeros((k, 1))
	clusterSize = np.zeros((k, 1))
	for i in range(k):
		pointsInClust = np.where(clusterAssignments == i)
		clusterSize[i] = pointsInClust.shape[0]
		pointCount = np.zeros((c, 1))
		for j in range(c):
			#### FIX THIS! pointCount[j] = np.sum(groundTruth[pointsInClust] == j)
		numPoints = np.amax(pointCount)
		assign[i] = np.argmax(pointCount)
		purity[i] = numPoints / clusterSize[i]
	averagePurity = np.dot(purity,clusterSize) / groundTruth.shape[0]
	return purity, clusterSize, averagePurity