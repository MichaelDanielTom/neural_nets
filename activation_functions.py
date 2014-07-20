"""
	Activation functions that take the weighted sum of inputs and return a 
	float in [0,1]
"""
THRESHOLD_ACTIVATION = 5


def hard_activation(in_j):
	return in_j > 5
