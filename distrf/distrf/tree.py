#!/usr/bin/env python3


"""A Distributed Distributed Tree """


from mpi4py import MPI

import numpy as np
import pandas as pd
from statistics import mode
from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Node:
    def __init__(
        self,
        *,
        feature=None,
        data=None,
        branch=None,
        parent=None,
        leaf=False,
        children=[]
    ):
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
    def __init__(self, *, criterion=None):
        """Pre-pruning criterion = {max_depth, partition_threshold, low_gain}"""
        self.root = None
        self.levels = None
        self.criterion = criterion

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
        return self.entropy(X, y) - self.rem(X, y, d)

    def build_tree(self, X, y, *, parent=None, branch=None, depth=0):
        """Performs the ID3 algorithm"""
        if len(y.unique()) == 1:  # all instances have the same target feature values
            return Node(feature=y.iat[0],
                        data=pd.concat([X, y], axis=1),
                        branch=branch,
                        parent=parent,
                        leaf=True)
        elif X.empty:  # dataset is empty, return a leaf node labeled with the majority class of the parent
            return Node(feature=mode(parent.y),
                        branch=branch,
                        parent=parent,
                        leaf=True)
        elif all((X[d] == X[d].iloc[0]).all() for d in X.columns):  # if all feature values are identical
                return Node(feature=mode(y),
                            data=pd.concat([X, y], axis=1),
                            branch=branch,
                            parent=parent,
                            leaf=True)
        elif self.criterion.get("max_depth"):
            if depth >= self.criterion["max_depth"]:
                return Node(feature=mode(y),
                            data=pd.concat([X, y], axis=1),
                            branch=branch,
                            parent=parent,
                            leaf=True)
        elif self.criterion.get("partition_threshold"):
            if len(X) < self.criterion["partition_threshold"]:
                return Node(feature=mode(y),
                            data=pd.concat([X, y], axis=1),
                            branch=branch,
                            parent=parent,
                            leaf=True)

        gain = np.argmax([self.information_gain(X, y, d) for d in X.columns])

        if self.criterion.get('low_gain'):
            if gain <= self.criterion["low_gain"]:
                return Node(feature=mode(y),
                            data=pd.concat([X, y], axis=1),
                            branch=branch,
                            parent=parent,
                            leaf=True)

        best_feature = X.columns[gain]
        best_node = deepcopy(Node(feature=best_feature,
                                  data=pd.concat([X, y], axis=1),
                                  branch=branch,
                                  parent=parent))

        partitions = [self.partition(X, y, best_feature, level) for level in self.levels[best_feature]]

        for *d, level in partitions:
            best_node.children.append(self.build_tree(*d, parent=best_node, branch=level, depth=depth+1))

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

    def score(self, X):
        return [self.predict(X.iloc[x].to_frame().T) for x in range(len(X))]


def voter(*argv):
    if len(argv) == 1:
        return argv[0]
    for arg in argv:
        if arg is None: continue
        return list(map(lambda f: mode(f), list(zip(voter(arg)))))


MPI_MODE = MPI.Op.Create(voter, commute=True)


def sub_sample(X, n_sample=2):
    """Enforces feature randomness"""
    return np.random.choice(X.columns.to_numpy(), n_sample, replace=False)


def bootstrap_sample(X, y, n_sample, key=True):
    feature_subset = self.sub_sample(X, int(np.log2(len(X))))
    d = pd.concat([X, y], axis=1)
    d = d.sample(n=n_sample, replace=key)
    return d.iloc[:, :-1][feature_subset], d.iloc[:, -1]


if __name__ == '__main__':
    df = pd.read_pickle(r'/Users/duong-jason/Desktop/dataset/cancer.pkl')
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    comm.Barrier()
    start_time = MPI.Wtime()

    dt = DecisionTree(criterion={'max_depth': 4}).fit(*self.bootstrap_sample(X, y, self.n_sample))
    score = dt.score(X_test)

    comm.Barrier()
    y_hat = comm.reduce(score, op=MPI_MODE, root=0)

    end_time = MPI.Wtime()
    if not rank:
        print(f"Accuracy Score: {accuracy_score(y_test, y_hat)*100:.2f}%")
        print(f"Execution Time: {end_time-start_time:.3f}s")
