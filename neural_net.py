"""
	Utility functions for neural networks
	So far taken from pg 728-737 of Russell & Norvig


"""
from activation_functions import hard_activation
import numpy as np
import json
from ipdb import set_trace
from collections import defaultdict


class NeuralNetwork(object):
	def __init__(self, act_fn, nn_array):
		self.act_fn = act_fn
		
		edges = []
		nodes = {}

		for edge in nn_array:
			nn_edge = NeuralNetworkEdge(*edge)
			edges.append(nn_edge)
			nn_edge.sync_with_nodes(nodes, self.act_fn)

		self.edges = edges
		self.nodes = nodes

	@property
	def input_nodes(self):
		return [node for i, node in self.nodes.iteritems() if node.is_input()]

	@property
	def output_nodes(self):
		return [node for i, node in self.nodes.iteritems() if node.is_output()]

	def run(self, inputs):
		self.initialize_activations(inputs)
		set_trace()
		propogate(2)

	def initialize_activations(self, inputs):
		#inputs is dict with nn node indicies as keys and activations as values
		for i_node, activation in inputs.iteritems():
			self.nodes[i_node].activation = activation

	def propogate(self, i_node):
		#propogates TO i_node, assumes all in_edges to node[i_node] are activated
		node = self.nodes[i_node]
		node.activation = sum([edge.i_start.activation * edge.weight for edge
			in node.in_edges]) + node.bias
		# feature and weight vectors
		x_i = np.zeros_like(node.in_edges)
		w_i = np.zeros_like(node.in_edges)

		set_trace()
		for edge in node.in_edges:
			#x_i.append(edge.i_
			pass


class NeuralNetworkNode(object):
	def __init__(self, activation_function, in_edges=None, out_edges=None, bias=0):
		self.activation_function = activation_function
		self.in_edges = in_edges if in_edges is not None else []
		self.out_edges = out_edges if out_edges is not None else []
		self.bias = bias
	
	def add_in_edge(self, edge_id):
		# remember dummy input a_0
		self.in_edges.append(edge_id)

	def add_out_edge(self, edge_id):
		self.in_edges.append(edge_id)

	@property
	def is_input(self):
		return True if (self.out_edges and not self.in_edges) else False

	@property
	def is_output(self):
		return True if (self.in_edges and not self.out_edges) else False

	def __unicode__(self):
		return "In: {} Out: {}".format(self.in_edges, self.out_edges)


class NeuralNetworkEdge(object):
	def __init__(self, weight, i_start, i_end):
		self.weight = weight
		self.i_start = i_start
		self.i_end = i_end

	def __unicode__(self):
		return "weight: {} start: {} end: {}".format(self.weight,
												     self.i_start,
													 self.i_end)

	def sync_with_nodes(self, nodes, act_fn):
		# add input node if it doesn't exist
		if self.i_start in nodes:
			nodes[self.i_start].add_in_edge(self.i_start)
		else:
			nodes[self.i_start] = NeuralNetworkNode(act_fn, [], [self.i_end])

		# same with output node
		if self.i_end in nodes:
			nodes[self.i_end].add_out_edge(self.i_end)
		else:
			nodes[self.i_end] = NeuralNetworkNode(act_fn, [self.i_start], [])


if __name__ == '__main__':
	# format: [weight, start_id, end_id]
	# starting with -1 means input, ending with -1 means output
	nn = np.array(((1, 0, 2),
				  (1, 1, 2),
				  (1, 3, 2)))
	nn = NeuralNetwork(hard_activation, nn)
	ipdb.set_trace()
	nn.run()
