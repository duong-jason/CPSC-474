import numpy as np
import pandas as pd
from copy import deepcopy

from statistics import mode, mean
from sklearn.metrics import accuracy_score, mean_squared_error


class Node:
    """
    A Decision Tree Node

    Parameters
    ----------
    feature - the value of a descriptive/target feature of a node
    data - the partitioned dataset resulting from the parent node on a feature value
    branch - the feature value from the parent node
    parent - the immediate adjacent node along the path from the root
    leaf - denotes a terminal node whose prediction is based on the path from the root to the node
    depth - the number of levels from the root to a node
    children - the nodes resulting from each unique feature value of the parent
    """
    def __init__(
        self,
        *,
        feature=None,
        data=None,
        branch=None,
        parent=None,
        leaf=False,
        depth=0,
        children=[]
    ):
        self.feature = feature
        self.data = data
        self.branch = branch
        self.parent = parent
        self.leaf = leaf
        self.depth = depth
        self.children = children

    def __str__(self):
        return self.depth * '\t' + f"{self.feature} (Branch={self.branch})"

    @property
    def is_leaf(self):
        """Returns whether a node is terminal"""
        return self.leaf

    @property
    def X(self):
        """Returns the partitioned feature matrix of a node"""
        return self.data.iloc[:, :-1]

    @property
    def y(self):
        """Returns the partitioned target vector of a node"""
        return self.data.iloc[:, -1]


class DecisionTreeEstimator:
    """A Decision Tree Estimator"""
    def __init__(self, criterion={}):
        """
        Parameters
        ----------
        root: the starting node of the decision tree
        n_levels: contains a list of all unique feature values for each descriptive feature
        criterion (pre-pruning): {max_depth, partition_threshold, low_gain}
        """
        self.root = None
        self.n_levels = None
        self.criterion = criterion

    def __repr__(self, node=None):
        """Displays the decision tree (Pre-Order Traversal)"""
        if not node:
            node = self.root
        return str(node) + ''.join(['\n' + self.__repr__(child) for child in node.children])

    def partition(self, X, y, d, t):
        """Returns a subset of the training data with feature (d) with level (t)"""
        D = pd.concat([X.loc[X[d]==t], y.loc[X[d]==t]], axis=1)
        D = D.drop([d], axis=1)
        return D.iloc[:, :-1], D.iloc[:, -1], t

    def fit(self, X, y):
        self.n_levels = {d: X[d].unique() for d in X.columns}
        self.root = self.make_tree(X, y)
        return self

    def predict(self, x):
        node = self.root
        while not node.is_leaf:
            for child in node.children:
                if child.branch == x.get(node.feature).values:
                    node = child
                    return node
            else:
                raise ValueError(f"Branch {node.feature} -> {child.branch} does not exist")

    def score(self, X, y):
        return [self.predict(X.iloc[x].to_frame().T).feature for x in range(len(X))]