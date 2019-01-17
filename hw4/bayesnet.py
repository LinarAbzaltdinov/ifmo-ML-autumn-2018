import itertools
from copy import deepcopy
from collections import deque

from pgmpy.readwrite.BIF import BIFReader
import numpy as np


class BayesNet:

    def __init__(self, filename):
        reader = BIFReader(filename)
        values = reader.get_states()
        # запомнили переменные и их возможные (дискретные) значения
        self.variables = dict()
        for v_name in reader.get_variables():
            self.variables[v_name] = Variable(v_name, values.get(v_name))
        # заполнили структуру сети
        for e in reader.get_edges():
            v_from = self.variables.get(e[0])
            v_to = self.variables.get(e[1])
            v_from.add_child(v_to)
            v_to.add_parent(v_from)
        # заполнили вероятности
        probabilities = reader.get_values()
        for v_name, variable in self.variables.items():
            variable.set_probabilities(probabilities.get(v_name))

        self.elimination_order = []

    def calc_probability(self, var_name, evidence={}):
        variable = self.variables[var_name]

        evidence1 = dict()
        for v_name in evidence:
            evidence1[self.variables[v_name]] = evidence[v_name]
        evidence = evidence1  # keys as Variable object

        dependecies = [variable]
        queue = deque([variable])
        while len(queue) != 0:
            v = queue.popleft()
            for parent in v.parents:
                if parent not in dependecies:
                    dependecies.append(parent)
                    queue.append(parent)

        values_combinations = list(self.get_values_combinations(dependecies, evidence))
        res = 0
        for combination in values_combinations:
            prob_of_one_combination = 1
            for variable in dependecies:
                prob_of_one_combination *= variable.calc_cond_probability(combination)
            res += prob_of_one_combination
        return res

    def get_values_combinations(self, variables, evidence):

        if len(variables) == 0:
            return [evidence]

        variable = variables[0]
        if variable in evidence:
            variable_values = [evidence[variable]]
        else:
            variable_values = variable.values

        return (
            states
            for var_value in variable_values
            for states in self.get_values_combinations(variables[1:], {**evidence, variable: var_value})
        )


class Variable:

    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.parents = []
        self.children = []
        self.probabilities = []
        self.cond_probabilities = []

    def add_parent(self, variable):
        self.parents.append(variable)

    def add_child(self, child):
        self.children.append(child)

    def set_probabilities(self, probabilities):
        if self.isLeaf():
            self.set_full_probabilities(np.ravel(np.array(probabilities)))
        else:
            self.cond_probabilities = dict()

            parent_values_combinations = list(reversed(
                list(itertools.product(*list(map(lambda v: list(reversed(v.values)), self.parents))))))

            for i, value in enumerate(self.values):
                self.cond_probabilities[value] = dict()
                for j, parent_combination in enumerate(parent_values_combinations):
                    self.cond_probabilities[value][parent_combination] = probabilities[i, j]

    def set_full_probabilities(self, probabilities):
        self.probabilities = dict(zip(self.values, np.array(probabilities)))

    def calc_cond_probability(self, current_states):
        self_value = current_states[self]
        if self.isLeaf():
            return self.probabilities[self_value]
        parent_combination = tuple(map(lambda v: current_states[v], self.parents))
        return self.cond_probabilities[self_value][parent_combination]

    @property
    def get_probabilities(self):
        return self.probabilities

    @property
    def get_values(self):
        return self.values

    @property
    def get_children(self):
        return self.children

    @property
    def get_parents(self):
        return self.parents

    def isLeaf(self):
        return len(self.parents) == 0


def main():
    filename = 'child.bif'
    folder = '/Users/linarkou/Documents/ML-course/hw4/'
    net = BayesNet(folder + filename)
    print(net.variables)
    print(net.elimination_order)

    variable = net.variables['HypoxiaInO2']
    dependecies = [variable]
    queue = deque([variable])
    while len(queue) != 0:
        v = queue.popleft()
        for parent in v.parents:
            if parent not in dependecies:
                dependecies.append(parent)
                queue.append(parent)

    for c in list(net.get_values_combinations(dependecies, {})):
        print('---')
        for key in c:
            print(key.name, c[key])

    print(net.calc_probability('HypoxiaInO2', {'HypoxiaInO2': 'Mild'}))
    print(net.calc_probability('HypoxiaInO2', {'HypoxiaInO2': 'Moderate'}))
    print(net.calc_probability('HypoxiaInO2', {'HypoxiaInO2': 'Severe'}))

    # net.calc_full_probabilities()
    # for v_name, variable in net.variables.items():
    #     print(v_name)
    #     print(variable.probabilities)

if __name__ == '__main__':
    main()
