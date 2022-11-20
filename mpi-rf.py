#!/usr/bin/env python3

from mpi4py import MPI

import numpy as np
import pandas as pd
from statistics import mode
from copy import deepcopy

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from optbinning import OptimalBinning


pd.set_option('mode.chained_assignment', None)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Node:
    def __init__(self, feature=None, data=None, branch=None, parent=None, leaf=False, children=[]):
        self.feature = feature
        self.data = data
        self.branch = branch
        self.parent = parent
        self.leaf = leaf
        self.children = children

    @property
    def isLeaf(self):
        return self.leaf

    @property
    def X(self):
        return self.data.iloc[:, :-1]

    @property
    def y(self):
        return self.data.iloc[:, -1]



class DecisionTree:
    """A Rudimentary Decision Tree Classifier"""
    def __init__(self):
        self.root = None
        self.levels = None

    def __repr__(self, node=None, depth=0):
        """Displays the decision tree"""
        if not node:
            node = self.root

        print(depth * '\t', node.feature, f"(Branch={node.branch})")
        for child in node.children:
            self.__repr__(child, depth+1)

        return ""

    def partition(self, X, y, d, t):
        D = pd.concat([X.loc[X[d]==t], y.loc[X[d]==t]], axis=1)
        D = D.drop([d], axis=1)
        return D.iloc[:, :-1], D.iloc[:, -1], t

    def entropy(self, X, y):
        """Measures the amount of uncertainty/impurity/heterogeneity in (X, y)"""
        proba = lambda t: len(X.loc[y==t]) / len(X)
        return -sum([proba(t) * np.log2(proba(t)) for t in y.unique()])

    def rem(self, X, y, d):
        """Measures the entropy after partitioning (X, y) on feature (d)"""
        weight = lambda t: len(X.loc[X[d]==t]) / len(X)
        return sum([weight(t) * self.entropy(X.loc[X[d]==t], y.loc[X[d]==t]) for t in X[d].unique()])

    def information_gain(self, X, y, d):
        """Measures the reduction in the overall entropy in (X, y) achieved by testing on feature (d)"""
        if debug:
            print(f"{d} = {self.entropy(X, y):.3f} - {self.rem(X, y, d):.3f} = {self.entropy(X, y) - self.rem(X, y, d):.3f}") 

        return self.entropy(X, y) - self.rem(X, y, d)

    def build_tree(self, X, y, *, parent=None, branch=None):
        """Performs the ID3 algorithm"""
        if len(y.unique()) == 1:  # all instances have the same target feature values
            if debug:
                print("All instances have the same target feature value\n")
            best_node = Node(feature=y.iat[0],
                             data=pd.concat([X, y], axis=1),
                             branch=branch,
                             parent=parent,
                             leaf=True)
        elif X.empty:  # dataset is empty, return a leaf node labeled with the majority class of the parent
            if debug:
                print("Dataset is empty\n")
            best_node =  Node(feature=mode(parent.y),
                              branch=branch,
                              parent=parent,
                              leaf=True)
        elif all((X[d] == X[d].iloc[0]).all() for d in X.columns):  # if all feature values are identical
            if debug:
                print("All instances have the same descriptive features\n")
            best_node = Node(feature=mode(y),
                             data=pd.concat([X, y], axis=1),
                             branch=branch,
                             parent=parent,
                             leaf=True)

        else:
            if debug:
                print("===Information Gain===")
            best_feature = X.columns[np.argmax([self.information_gain(X, y, d) for d in X.columns])]
            best_node = deepcopy(Node(feature=best_feature,
                                 data=pd.concat([X, y], axis=1),
                                 branch=branch,
                                 parent=parent))

            if debug:
                print()
                print("===Best Feature===")
                print(best_feature)
                print()

            partitions = [self.partition(X, y, best_feature, t) for t in self.levels[best_feature]]

            for *d, t in partitions:
                if debug:
                    print(f"===Partitioned Dataset ({t})===")
                    print(pd.concat([*d], axis=1).head())
                    print()
                best_node.children.append(self.build_tree(*d, parent=best_node, branch=t))
        return best_node

    def fit(self, X, y):
        self.levels = {k: X[k].unique() for k in X.columns}
        self.root = self.build_tree(X, y)
        return self

    def predict(self, x):
        node = self.root
        while not node.isLeaf:
            for child in node.children:
                if child.branch == x.get(node.feature).values:
                    node = child
                    break
        return node.feature

    def score(self, X, y):
        y_hat = [self.predict(X.iloc[x].to_frame().T) for x in range(len(X))]
        return accuracy_score(y, y_hat)


class RandomForest:
    def __init__(self, n_estimators=5, n_sample=2):
        self.n_estimators = n_estimators
        self.n_sample = n_sample
        self.tree = None

    def __repr__(self):
        for i, dt in enumerate(self.forest, start=1):
            print(f"===Decision Tree  #{i}===")
            print(dt)
        print()
        return ""

    def subsample(self, X, n_sample=2):
        """Enforces feature randomness"""
        return np.random.choice(X.columns.to_numpy(), n_sample, replace=False)

    def make_bootstrap(self, X, y, n_sample, key=True):
        feature_subset = self.subsample(X, int(np.log2(len(X))))
        d = pd.concat([X, y], axis=1)
        d = d.sample(n=n_sample, replace=key)
        return d.iloc[:, :-1][feature_subset], d.iloc[:, -1]

    def fit(self, X, y):
        self.tree = DecisionTree().fit(*self.make_bootstrap(X, y, self.n_sample))
        return self

    def predict(self, x):
        """Aggregation"""
        # assert all(isinstance(model, DecisionTree) for model in self.forest)

        pred = [None] * size

        if not rank:
            print("Process 0 is now scattering")
        pred = comm.scatter(pred, root=0)

        print(f"Process {rank} is predicting")
        # each process constitutes onepred 
        pred = self.tree.predict(x)
        print(f"Process {rank} predicted {pred}")

        comm.Barrier()

        if rank == 0:
            print("All Process is done predicting")
            print(f"Process 0 is now gathering")

        votes = comm.allgather(pred)

        print(f"Rank {rank} has Votes={votes}")

        return mode(votes)
        # return mode([dt.predict(X) for dt in self.forest])

    def score(self, X, y):
        # for each test sample, all trees decide on the final prediction
        y_hat = self.predict(X.iloc[0].to_frame().T)
        # y_hat = [self.predict(X.iloc[x].to_frame().T) for x in range(len(X))]
        # if rank == 0:
        #     return accuracy_score(y, y_hat)
        return y_hat


if __name__ == '__main__':
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    debug = 0

    # Source: http://gnpalencia.org/optbinning/tutorials/tutorial_binary.html
    for d in X.columns:
        op = OptimalBinning(name=d, dtype="numerical", solver="cp")
        op.fit(X[d].values, y)
        bins = [0] + list(op.splits) + [X[d].max()]
        X[d] = pd.cut(X[d], bins, labels=range(len(bins)-1))

    for d in ['mean concavity', 'mean concave points', 'concavity error', 'concave points error', 'worst concavity', 'worst concave points']:
        X[d].fillna(mode(X[d].values), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    comm.Barrier()
    start_time = MPI.Wtime()

    rf = RandomForest(n_sample=len(X_train)).fit(X_train, y_train)
    rf.score(X_test, y_test)
    # print(rf.score(X_test, y_test))

    end_time = MPI.Wtime()
    if rank == 0:
        print("Execution Time:", end_time-start_time)
