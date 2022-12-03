from .base_estimator import *

from copy import deepcopy
from statistics import mode, mean
from sklearn.metrics import accuracy_score, mean_squared_error


class DecisionTreeClassifier(DecisionTreeEstimator):
    """A Rudimentary Decision Tree Classifier"""

    def __init__(self, *, metric="entropy", eval="info_gain", criterion={}):
        """
        Metric
            - {gain, gini}

        Eval
            - {info_gain, gain_ratio}
        """
        super().__init__(criterion)
        self.metric = self.entropy if metric == "entropy" else self.gini
        self.eval = (
            self.information_gain
            if eval == "info_gain"
            else self.information_gain_ratio
        )

    def gini(self, X, y):
        proba = lambda t: len(X.loc[y == t]) / len(X)
        return 1 - np.sum([proba(t) ** 2 for t in y.unique()])

    def entropy(self, X, y):
        """
        Measures the amount of uncertainty/impurity/heterogeneity in (X, y)
        """
        proba = lambda t: len(X.loc[y == t]) / len(X)
        return -np.sum([proba(t) * np.log2(proba(t)) for t in y.unique()])

    def rem(self, X, y, d):
        """
        Measures the entropy after partitioning (X, y) on feature (d)
        """
        weight = lambda t: len(X.loc[X[d] == t]) / len(X)
        return np.sum([weight(t) * self.metric(X.loc[X[d] == t], y.loc[X[d] == t])
                       for t in X[d].unique()])

    def information_gain(self, X, y, d):
        """
        Measures the reduction in the overall entropy in (X, y) achieved by testing on feature (d)
        """
        if is_numeric_dtype(X[d]):
            df = pd.concat([X, y], axis=1)
            df.sort_values(by=[d], inplace=True)
            gain, optimal_threshold = self.find_optimal_threshold(df, d)

            self.optimal_threshold[d] = optimal_threshold

            return gain

        return self.metric(X, y) - self.rem(X, y, d)

    def information_gain_ratio(self, X, y, d):
        proba = lambda t: len(X.loc[X[d] == t]) / len(X)
        entropy = lambda: -np.sum([proba(t) * np.log2(proba(t)) for t in X[d].unique()])
        return self.metric(X, y) - self.rem(X, y, d) / entropy()

    def make_tree(self, X, y, *, parent=None, branch=None, depth=0):
        """Performs the ID3 algorithm

        Base Cases
        ----------
        - all instances have the same target feature values
        - dataset is empty, return a leaf node labeled with the majority class of the parent
        - if all feature values are identical
        - max_depth reached
        - max number of instances in partitioned dataset reached
        """
        make_node = lambda f, t: Node(
            feature=f,
            data=pd.concat([X, y], axis=1),
            branch=branch,
            parent=parent,
            depth=depth,
            leaf=t,
        )

        if len(y.unique()) == 1:
            return make_node(y.iat[0], True)
        elif X.empty:
            return make_node(mode(parent.y), True)
        elif all((X[d] == X[d].iloc[0]).all() for d in X.columns):
            return make_node(mode(y), True)
        if self.criterion.get("max_depth", float("inf")) <= depth:
            return make_node(mode(y), True)
        if self.criterion.get("partition_threshold", float("-inf")) >= len(X):
            return make_node(mode(y), True)

        max_gain = np.argmax([self.eval(X, y, d) for d in X.columns])

        if self.criterion.get("low_gain", float("-inf")) >= max_gain:
            return make_node(mode(y), True)

        best_feature = X.columns[max_gain]
        best_node = deepcopy(make_node(best_feature, False))

        if is_numeric_dtype(X[best_feature]):
            eps = self.optimal_threshold[best_feature]
            left = [X.loc[X[best_feature] < eps], y.loc[X[best_feature] < eps], eps]
            right = [X.loc[X[best_feature] >= eps], y.loc[X[best_feature] >= eps], eps]
            X_levels = [left, right]
        else:
            X_levels = [
                self.partition(X, y, best_feature, level)
                for level in self.n_levels[best_feature]
            ]

        for *d, level in X_levels:
            best_node.children.append(
                self.make_tree(*d, parent=best_node, branch=level, depth=depth + 1)
            )
        return best_node

    def score(self, X, y):
        y_hat = super().score(X, y)
        return accuracy_score(y, y_hat)


class DecisionTreeRegressor(DecisionTreeEstimator):
    """A Rudimentary Decision Tree Regressor"""

    def __init__(self, *, criterion={}):
        super().__init__(criterion)
        self.metric = self.variance

    def variance(self, X, y):
        if len(X) == 1:
            return 0
        return np.sum([(t - mean(y)) ** 2 for t in y]) / (len(X) - 1)

    def weighted_variance(self, X, y, d):
        if is_numeric_dtype(X[d]):
            df = pd.concat([X, y], axis=1)
            df.sort_values(by=[d], inplace=True)
            gain, optimal_threshold = self.find_optimal_threshold(df, d)

            self.optimal_threshold[d] = optimal_threshold

            return gain

        weight = lambda t: len(X.loc[X[d] == t]) / len(X)
        return np.sum([weight(t) * self.metric(X.loc[X[d] == t], y.loc[X[d] == t])
                       for t in X[d].unique()])

    def make_tree(self, X, y, *, parent=None, branch=None, depth=0):
        """
        Performs the ID3 algorithm

        Base Cases
        ----------
        - all instances have the same target feature values
        - dataset is empty, return a leaf node labeled with the majority class
        - max_depth reached
        - max number of instances in partitioned dataset reached
        """
        make_node = lambda f, t: Node(
            feature=f,
            data=pd.concat([X, y], axis=1),
            branch=branch,
            parent=parent,
            depth=depth,
            leaf=t,
        )

        if len(y.unique()) == 1:
            return make_node(y.iat[0], True)
        elif X.empty:
            return make_node(mean(y), True)
        if self.criterion.get("max_depth", float("inf")) <= depth:
            return make_node(mean(y), True)
        if self.criterion.get("partition_threshold", float("-inf")) >= len(X):
            return make_node(mean(y), True)

        min_var = np.argmin([self.weighted_variance(X, y, d) for d in X.columns])

        best_feature = X.columns[min_var]
        best_node = deepcopy(make_node(best_feature, False))

        if is_numeric_dtype(X[best_feature]):
            eps = self.optimal_threshold[best_feature]
            left = [X.loc[X[best_feature] < eps], y.loc[X[best_feature] < eps], eps]
            right = [X.loc[X[best_feature] >= eps], y.loc[X[best_feature] >= eps], eps]
            X_levels = [left, right]
        else:
            X_levels = [
                self.partition(X, y, best_feature, level)
                for level in self.n_levels[best_feature]
            ]

        for *d, level in X_levels:
            best_node.children.append(
                self.make_tree(*d, parent=best_node, branch=level, depth=depth + 1)
            )
        return best_node

    def score(self, X, y):
        y_hat = super().score(X, y)
        return mean_squared_error(y, y_hat, squared=False)
