"""We are going to have input nodes
output nodes
global default graph variable
and a compute method"""
import numpy as np

from Project1 import graph


class Operation():
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        graph._default_graph.operations.append(self)
        for node in input_nodes:
            node.output_nodes.append(self)  # once the operation is done, add it to the output nodes

    def compute(self):
        pass


class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        self.inputs = [x, y]
        return x + y


class multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        self.inputs = [x, y]
        return x * y


class matmul(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        self.inputs = [x, y]
        return x.dot(y)  # numpy operation


class Sigmoid(Operation):
    def __init__(self, z):
        super().__init__(z)

    def compute(self, z):
        return 1 / (1 + np.exp(-z))
