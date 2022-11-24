from .base_estimator import *


class DecisionTreeRegressor(DecisionTreeEstimator):
    """A Rudimentary Decision Tree Regressor"""
    def __init__(self, *, criterion={}):
        super().__init__(criterion)

    def variance(self, X, y):
        if len(X) == 1:
            return 0
        return np.sum([(t-mean(y))**2 for t in y]) / (len(X)-1)

    def weighted_variance(self, X, y, d):
        weight = lambda t: len(X.loc[X[d]==t]) / len(X)
        return np.sum([weight(t) * self.variance(X.loc[X[d]==t], y.loc[X[d]==t]) for t in X[d].unique()])

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
        make_node = lambda f, t: Node(feature=f, data=pd.concat([X, y], axis=1), branch=branch, parent=parent, depth=depth, leaf=t)

        if len(y.unique()) == 1:
            return make_node(y.iat[0], True)
        elif X.empty:
            return make_node(mean(y), True)
        elif self.criterion.get("max_depth", float('inf')) <= depth:
            return make_node(mean(y), True)
        elif self.criterion.get("partition_threshold", float('-inf')) >= len(X):
            return make_node(mean(y), True)

        min_var = np.argmin([self.weighted_variance(X, y, d) for d in X.columns])

        best_feature = X.columns[min_var]
        best_node = deepcopy(make_node(best_feature, False))

        X_levels = [self.partition(X, y, best_feature, level) for level in self.n_levels[best_feature]]

        for *d, level in X_levels:
            best_node.children.append(self.make_tree(*d, parent=best_node, branch=level, depth=depth+1))
        return best_node

    def score(self, X, y):
        y_hat = super().score(X, y)
        return mean_squared_error(y, y_hat, squared=False)

if __name__ == '__main__':
    data = {
        'Season': ['winter', 'winter', 'winter', 'spring', 'spring', 'spring', 'summer', 'summer', 'summer', 'autumn', 'autumn', 'autumn'],
        'Work Day': ['false', 'false', 'true', 'false', 'true', 'true', 'false', 'true', 'true', 'false', 'false', 'true'],
        'Rentals': [800, 826, 900, 2100, 4740, 4900, 3000, 5800, 6200, 2910, 2880, 2820],
    }

    df = pd.DataFrame(data)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    dt_regr = DecisionTreeRegressor().fit(X, y)
    print(dt_regr)