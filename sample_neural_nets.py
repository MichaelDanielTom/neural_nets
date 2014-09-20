""" 
    contains sample serialized neural networks with associated inputs
    format: [weight, start_id, end_id]
"""
import json
import numpy as np

# input nodes: 0,1,3
# output nodes: 2
SIMPLE_NN = {'nn': np.array(((1, 0, 2), (1, 1, 2), (1, 3, 2))),
             'inputs': {0: 2, 1: 3, 3: 4}}

# weights initialized to 0
# note the 0th node is always the bias and activated to 1
AND_NN = {'nn': np.array(((0, 0, 3), (0, 1, 3), (0, 2, 3))),
          'training': [
              {'inputs': {0: 1, 1: 1, 2: 1}, 'output': 1},
              {'inputs': {0: 1, 1: 0, 2: 0}, 'output': 0},
              {'inputs': {0: 1, 1: 1, 2: 0}, 'output': 0},
              {'inputs': {0: 1, 1: 0, 2: 1}, 'output': 0}
          ]
}

XOR_NN = {'nn': np.array(((0, 0, 3), (0, 1, 3), (0, 2, 3))),
          'training': [
              {'inputs': {0: 1, 1: 1, 2: 1}, 'output': 0},
              {'inputs': {0: 1, 1: 0, 2: 0}, 'output': 1},
              {'inputs': {0: 1, 1: 1, 2: 0}, 'output': 1},
              {'inputs': {0: 1, 1: 0, 2: 1}, 'output': 0}
          ]
}
