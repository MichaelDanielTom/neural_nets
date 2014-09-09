"""
	Utility functions for neural networks
	So far taken from pg 728-737 of Russell & Norvig
"""
from __future__ import division
import json
import random
import numpy as np
from pprint import pprint
from ipdb import set_trace
from collections import defaultdict

from sample_neural_nets import *
from activation_functions import hard_activation


class NeuralNetwork(object):
	def __init__(self, act_fn, nn_array):
		self.act_fn = act_fn
		
		edges = []
		nodes = {}

		for edge in nn_array:
			nn_edge = NeuralNetworkEdge(*edge)
			edges.append(nn_edge)
			edge_index = len(edges) - 1

			if nn_edge.i_start in nodes:
				nodes[nn_edge.i_start].add_out_edge(edge_index)
			else:
				nodes[nn_edge.i_start] = NeuralNetworkNode(act_fn, [], [edge_index])

			if nn_edge.i_end in nodes:
				nodes[nn_edge.i_end].add_in_edge(edge_index)
			else:
				nodes[nn_edge.i_end] = NeuralNetworkNode(act_fn, [edge_index], [])

		self.edges = edges
		self.nodes = nodes

	@property
	def input_nodes(self):
		return [node for node in self.nodes.values() if node.is_input]

	@property
	def output_nodes(self):
		return [node for node in self.nodes.values() if node.is_output]

	@property
	def output_i_nodes(self):
		return [i_node for i_node, node in self.nodes.iteritems() if node.is_output]

	def run(self, inputs):
		self.initialize_activations(inputs)

		#BFS to propogate to nodes
		i_nodes = inputs.keys()
		for i_node_layer in self._i_node_layer_iter(inputs.keys()):
			for i_node in i_node_layer:
				self.propogate(i_node)

		return {i_node: self.nodes[i_node].activation for i_node in
				self.output_i_nodes}

	def _i_node_layer_iter(self, start_nodes):
		# takes a set of starting nodes and iteratively yields each layer 
		# forward of this layer till the end of the NN
		while start_nodes:
			start_nodes = self._get_out_nodes(start_nodes)
			if start_nodes:
				yield start_nodes

	def _get_out_nodes(self, i_nodes):
		# takes list of node indicies and returns set of output nodes
		tot_out_nodes = set()
		for i_node in i_nodes:
			out_nodes = [self.edges[i_edge].i_end for i_edge in
						 self.nodes[i_node].out_edges]
			tot_out_nodes.update(out_nodes)
		return tot_out_nodes

	def initialize_activations(self, inputs):
		#inputs is dict with nn node indicies as keys and activations as values
		for i_node, activation in inputs.iteritems():
			self.nodes[i_node].activation = activation

	def propogate(self, i_node):
		#propogates TO i_node, assumes all in_edges to node[i_node] are activated
		node = self.nodes[i_node]
		weighted_sum = 0
		for i_edge in node.in_edges:
			in_edge = self.edges[i_edge]
			in_node = self.nodes[self.edges[i_edge].i_start]
			weighted_sum += in_node.activation * in_edge.weight

		node.activation = node.activation_function(weighted_sum + node.bias)


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


def run_and_nn():
	nn = NeuralNetwork(hard_activation, AND_NN['nn'])
	single_layer_perceptron_learn(nn, AND_NN['training'])

	# test that all training data passes
	for datum in AND_NN['training']:
		assert(data_passes(nn, datum))
	print "we trained it!"


def data_passes(nn, datum):
	actual_outputs = nn.run(datum['inputs'])
	output = actual_outputs.values()[0]
	passes = True if output == datum['output'] else False
	return passes


def single_layer_perceptron_learn(nn, training_data, num_iter=10):
	"""
	perceptron learning single layer: 
		initialize weights to 0
		for each training example, run the inputs through the nn, and update 
		the weights to w_i = w_i + a(expected - actual)x_i
			a is the learning rate
			x_i is the input for that node
		repeat
	"""
	learning_rate = 0.2 
	for i in xrange(num_iter):
		for datum in random.sample(training_data, len(training_data)):
			inputs = datum['inputs']
			expected_out = datum['output']
			output_activations = nn.run(inputs)

			# for now only single output
			output = output_activations.values()[0]
			print [edge.weight for edge in nn.edges]
			for edge in nn.edges:
				in_activation = nn.nodes[edge.i_start].activation
				edge.weight += learning_rate*(expected_out-output)*in_activation


if __name__ == '__main__':
	run_and_nn()
	if 0:
		nn = NeuralNetwork(hard_activation, SIMPLE_NN['nn'])
		output_activations = nn.run(SIMPLE_NN['inputs'])
		pprint(output_activations)
