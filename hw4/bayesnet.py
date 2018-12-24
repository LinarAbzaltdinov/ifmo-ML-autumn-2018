from copy import deepcopy

from pgmpy.readwrite.BIF import BIFReader


class BayesNet:

    def __init__(self, filename):
        reader = BIFReader(filename)
        values = reader.get_states()
        self.variables = dict()
        for v in reader.get_variables():
            self.variables[v] = Variable(v, values.get(v))
        for e in reader.get_edges():
            v_from = self.variables.get(e[0])
            v_to = self.variables.get(e[1])
            v_from.add_child(v_to)
            v_to.add_parent(v_from)
        probabilities = reader.get_values()
        for v_name, variable in self.variables.items():
            variable.set_probabilities(probabilities.get(v_name))
        self.elimination_order = []
        self.__eval_elimination_order()

    def __eval_elimination_order(self):
        variables = deepcopy(self.variables)
        while len(variables) != 0:
            to_delete = set()
            curr_vars = set(variables.keys())
            for v_name, variable in variables.items():
                if len(variable.parents) == 0 \
                        or not any(x.name in curr_vars.difference(to_delete) for x in variable.parents):
                    self.elimination_order.append(v_name)
                    to_delete.add(v_name)
            for v_name in to_delete:
                del variables[v_name]

class Variable:

    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.parents = set()
        self.children = set()
        self.probabilities = []

    def add_parent(self, variable):
        self.parents.add(variable)

    def add_child(self, child):
        self.children.add(child)

    def set_probabilities(self, probabilities):
        self.probabilities = probabilities

    @property
    def get_children(self):
        return self.children

    @property
    def get_parents(self):
        return self.parents


def main():
    filename = 'child.bif'
    folder = '/Users/linarkou/Documents/ML-course/hw4/'
    net = BayesNet(folder + filename)
    print(net.variables)
    print(net.elimination_order)

if __name__ == '__main__':
    main()
