#!/usr/bin/env python3

from mpi4py import MPI

import numpy as np
import pandas as pd
import statistics as stat
from copy import deepcopy


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class Node:
    def __init__(
        self,
        feature=None,
        data=None,
        arc=None,
        leaf=False,
        parent=None,
        children=[]
    ):
        self.feature = feature
        self.data = data
        self.arc = arc
        self.leaf = leaf
        self.parent = parent
        self.children = children

    @property
    def isLeaf(self):
        return self.leaf

    @property
    def y_data(self):
        return self.data.iloc[:, -1]

    @property
    def X_data(self):
        return self.data.iloc[:, :-1]


class DecisionTree:
    def __init__(self):
        self.root = None

    def display_tree(self, node=None, depth=''):
        """Displays the decision tree"""
        if not depth: node = self.root

        print(depth, node.feature)
        if node:
            for child in node.children:
                self.display_tree(child, depth+'\t')

    def partition(self, X, y, d, t):
        p = pd.concat([X.loc[X[d]==t], y.loc[X[d]==t]], axis=1)
        p = p.drop([d], axis=1)
        return p.iloc[:, :-1], p.iloc[:, -1], t

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

    def fit(self, X, y, *, parent=None, level=None):
        """Performs the ID3 algorithm"""
        if len(y.unique()) == 1:  # all instances have the same target feature values
            return Node(feature=y.iloc[0],
                        data=pd.concat([X, y], axis=1),
                        arc=level,
                        leaf=True,
                        parent=parent)
        elif all((X[d] == X[d].iloc[0]).all() for d in X.columns):  # if all feature values are identical
            return Node(feature=stat.mode(y),
                        data=pd.concat([X, y], axis=1),
                        arc=level,
                        leaf=True,
                        parent=parent)
        elif X.empty:  # dataset is empty, return a leaf node labeled with the majority class of the parent
            return Node(feature=stat.mode(parent.y_data),
                        arc=level,
                        leaf=True,
                        parent=parent)

        if rank < len(X.columns):
            max_gain = 0
            max_gain = self.information_gain(X, y, X.columns[rank])

            print(f"Rank {rank} on {X.columns[rank]} = {max_gain}")

            comm.Barrier()
            comm.allreduce(max_gain, MPI.MAX)

            if rank == 0:
                print(max_gain)
                # best_feature = X.columns[max_gain]
                # print(best_feature)

        best_node = deepcopy(Node(feature=best_feature,
                             data=pd.concat([X, y], axis=1),
                             arc=level,
                             parent=parent))


        partitions = [self.partition(X, y, best_feature, t) for t in X[best_feature].unique()]

        if not self.root:
            self.root = best_node

        for *d, t in partitions:
            best_node.children.append(self.fit(*d, parent=best_node, level=t))

        if self.root is best_node:
            return self
        return best_node

    def predict(self, X):
        node = self.root
        while not node.isLeaf:
            arc = X[node.feature].iloc[0]
            for child in node.children:
                if arc == child.arc:
                    node = child
                    break
        return node.feature


if __name__ == '__main__':
    data = {
        'Stream': ['false', 'true', 'true', 'false', 'false', 'true', 'true'],
        'Slope': ['steep', 'moderate', 'steep', 'steep', 'flat', 'steep', 'steep'],
        'Elevation': ['high', 'low', 'medium', 'medium', 'high', 'highest', 'high'],
        'Vegetation': ['chapparal', 'riparian', 'riparian', 'chapparal', 'conifer', 'conifer', 'chapparal']}


    df = pd.DataFrame(data)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    dt = DecisionTree().fit(X, y)
