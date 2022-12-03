from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass(kw_only=True)
class Node:
    """
    A Decision Tree Node

    Parameters
    ----------
    feature
        - the value of a descriptive/target feature of a node

    data
        - the partitioned dataset resulting from the parent node on a feature value

    branch
        - the feature value from the parent node

    parent
        - the immediate adjacent node along the path from the root

    leaf
        - denotes a terminal node whose prediction is based on the path from the root to the node

    depth
        - the number of levels from the root to a node

    children
        - the nodes resulting from each unique feature value of the parent
    """

    feature: ...
    data: pd.DataFrame.dtypes
    branch: ...
    parent: Node
    depth: int
    leaf: bool
    children: list[Node] = field(default_factory=list)

    def __str__(self):
        return self.depth * "\t" + f"{self.feature} (Branch: {self.branch})"

    @property
    def is_leaf(self):
        """
        Returns whether a node is terminal
        """
        return self.leaf

    @property
    def X(self):
        """
        Returns the partitioned feature matrix of a node
        """
        return self.data.iloc[:, :-1]

    @property
    def y(self):
        """
        Returns the partitioned target vector of a node
        """
        return self.data.iloc[:, -1]

    @property
    def left(self):
        """
        Returns the left child
        """
        return self.children[0]

    @property
    def right(self):
        """
        Returns the right child
        """
        return self.children[1]


class DecisionTreeEstimator:
    """A Decision Tree Estimator"""

    def __init__(self, criterion={}):
        """
        Parameters
        ----------
        root
            - the starting node of the decision tree

        n_levels
            - contains a list of all unique feature values for each descriptive feature

        criterion (pre-pruning)
            - {max_depth, partition_threshold, low_gain}
        """
        self.root = None
        self.n_levels = None
        self.criterion = criterion
        self.optimal_threshold = {}

    def __repr__(self, node=None):
        """
        Displays the decision tree (Pre-Order Traversal)
        """
        if not node:
            node = self.root
        return str(node) + "".join(
            ["\n" + self.__repr__(child) for child in node.children]
        )

    def find_optimal_threshold(self, df, d):
        thresholds = []
        for i in range(len(df) - 1):
            pairs = df.iloc[i : i+2, -1]
            if any(pairs.iloc[0] != val for val in pairs.values):
                thresholds.append(df.loc[pairs.index, d].mean())

        levels = []
        for threshold in thresholds:
            X_a, X_b = df.loc[df[d] < threshold], df.loc[df[d] >= threshold]

            weight_left = len(X_a.loc[X_a[d] < threshold]) / len(df)
            weight_right = len(X_b.loc[X_b[d] >= threshold]) / len(df)

            metric = self.metric(df.iloc[:, :-1], df.iloc[:, -1])
            metric_left = self.metric(X_a.iloc[:, :-1], X_a.iloc[:, -1])
            metric_right = self.metric(X_b.iloc[:, :-1], X_b.iloc[:, -1])

            rem = metric_left * weight_left + metric_right * weight_right

            levels.append(metric - rem)

        return max(levels), thresholds[np.argmax(levels)]

    def partition(self, X, y, d, t):
        """
        Returns a subset of the training data with feature (d) with level (t)
        """
        D = pd.concat([X.loc[X[d] == t], y.loc[X[d] == t]], axis=1)
        D = D.drop([d], axis=1)
        return D.iloc[:, :-1], D.iloc[:, -1], t

    def fit(self, X, y):
        self.n_levels = {d: X[d].unique() for d in X.columns}
        self.root = self.make_tree(X, y)
        return self

    def predict(self, x):
        node = self.root
        while not node.is_leaf:
            query_branch = x.get(node.feature).values
            if is_numeric_dtype(query_branch):
                node = node.left if node.left.branch < query_branch else node.right
            else:
                for child in node.children:
                    if child.branch == query_branch:
                        node = child
                        break
                else:
                    raise ValueError(
                        f"Branch {node.feature} -> {x.get(node.feature).values} does not exist"
                    )
        return node

    def score(self, X, y):
        return [self.predict(X.iloc[x].to_frame().T).feature for x in range(len(X))]
