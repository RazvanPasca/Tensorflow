

class Placeholder:
    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)


class Variable:
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)


class Graph:
    def __init__(self):
        self.operations = []
        self.placeholders = []  # places for data to come
        self.variables = []  # changeable params in the graph

    def set_as_default(self):
        global _default_graph  # allows to access the var in other classes
        _default_graph = self

