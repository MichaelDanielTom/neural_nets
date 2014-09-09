"""
	Activation functions that take the weighted sum of inputs and return a 
	float in [0,1]
"""
THRESHOLD_ACTIVATION = 0


def hard_activation(in_j):
	if in_j > THRESHOLD_ACTIVATION:
		to_return = 1
	else:
		to_return = 0
	return to_return
