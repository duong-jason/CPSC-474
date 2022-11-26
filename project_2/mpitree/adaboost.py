from .classifier import *


class AdaBoostClassifier:
    """An AdaBoost Classifier"""
    def __init__(
        self,
        *,
        base_model=None,
        n_estimators=100,
        learning_rate=5e-2,
        criterion={}
    ):
        """
        Parameters
        ----------
        """
        self.base_model = DecisionTreeClassifier(criterion=criterion)

        self.learning_rate = learning_rate

    def fit(self, X, y):
        """"""
        pass

    def predict(self, x):
        """"""
        pass

if __name__ == '__main__':
    pass