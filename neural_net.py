"""
	Utility functions for neural networks
	So far taken from pg 728-737 of Russell & Norvig
"""
from activation_functions import hard_activation
import numpy as np
import json
import pdb
from collections import defaultdict

class NeuralNetwork(object):
	"""
		collection of nodes
		how to only save edge weight once. Relational database?
	"""
	def __init__(self, act_fn, nn_array):
		self.act_fn = act_fn
		
		edges = []
		nodes = {}

		for edge in nn_array:
			nn_edge = NeuralNetworkEdge(*edge)
			edges.append(nn_edge)

			# negative node indicies in the nn_array mean...

			# add input node if it doesn't exist
			if edge[1] in nodes:
				nodes[edge[1]].add_out_edge(edge[2])
			else:
				nodes[edge[1]] = NeuralNetworkNode(self.act_fn, [], [edge[2]])

			# same with output node
			if edge[2] in nodes:
				nodes[edge[2]].add_out_edge(edge[1])
			else:
				nodes[edge[2]] = NeuralNetworkNode(self.act_fn, [edge[1]], [])

		self.edges = edges
		self.nodes = nodes
	
	def run(self, inputs):
		NeuralNetworkNode
		outputs = 1
		return outputs 

	def get_input_nodes(self):
		# returns list of nodes that 
		pass
	
	@property
	def depth(self):
		return 1

class NeuralNetworkNode(object):
	def __init__(self, activation_function, input_nodes, output_nodes):
		self.activation_function = activation_function
		self.in_edges = [] 
		self.out_edges = [] 
	
	def add_in_edge(self, edge_id):
		# remember dummy input a_0
		self.in_edges.append(edge_id)

	def add_out_edge(self, edge_id):
		# remember dummy input a_0
		self.in_edges.append(edge_id)


class NeuralNetworkEdge(object):
	def __init__(self, weight, start, end):
		# edge with weight and start/end
		self.weight = weight
		self.start = start
		self.end = end
		

if __name__ == '__main__':
	# format: [weight, start_id, end_id]
	# starting with -1 means input, ending with -1 means output
	nn = np.array(((1, -1, 0),
				  (1, -1, 1),
				  (1, 0, -1)))
	with open('sample_nn.json', 'w') as f:
		nn.tofile(f)
	nn = NeuralNetwork(hard_activation, nn)
	pdb.set_trace()
