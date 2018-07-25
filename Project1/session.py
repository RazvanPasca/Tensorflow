from Project1.graph import *
from Project1.operation import *
import numpy as np

"""We go along the route to find the operations which need to be done before
E.g.: Ax+b has 2 nodes : y + b and ax = y. 
We want to make sure we do Ax first, then y + b"""


def traverse_postorder(operation):
    nodes_postorder = []

    def traverse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                traverse(input_node)
        nodes_postorder.append(node)

    traverse(operation)
    return nodes_postorder


class Session():
    """operation is the operation we want to compute
    feed_dict is to feed the data"""

    def run(self, operation, feed_dict={}):
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:

            if type(node) == Placeholder:
                node.output = feed_dict[node]

            elif type(node) == Variable:
                node.output = node.value

            else:
                #Operation
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output
