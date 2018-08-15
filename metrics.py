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

