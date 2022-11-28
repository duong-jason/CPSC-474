from mpi4py import MPI
from .decision_tree import *


"""A Distributed Random Forest Algorithm"""


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def voter(*argv):
    # # https://stackoverflow.com/questions/6208367/regex-to-match-stuff-between-parentheses
    # import re
    # if not rank:
    #     print(str(argv))
    #     # https://stackoverflow.com/questions/2403122/regular-expression-to-extract-text-between-square-brackets
    #     a = re.findall(r'\[([^\)]+)\]', str(argv))
    #     b = [i for i in [j for j in a] if i.isdigit()]
    #     print(a)
    #     quit()
    if len(argv) == 1:
        return argv[0]
    for arg in argv:
        if arg is None: continue
        return list(map(lambda f: mean(f), list(zip(voter(arg)))))


MPI_MODE = MPI.Op.Create(voter, commute=True)

class RandomForest(DecisionTreeClassifier, DecisionTreeRegressor):
    def __init__(self, n_sample=0, criterion={}):
        self.n_sample = n_sample
        self.criterion = criterion
        self.tree = None

    def sub_sample(self, X, n_sample=2):
        """
        Enforces feature randomness
        """
        return np.random.choice(X.columns.to_numpy(), n_sample, replace=False)

    def bootstrap_sample(self, X, y, n_sample, key=True):
        feature_subset = self.sub_sample(X, int(np.log2(len(X.columns))))
        d = pd.concat([X, y], axis=1)
        d = d.sample(n=n_sample, replace=key)
        return d.iloc[:, :-1][feature_subset], d.iloc[:, -1]

    def fit(self, X, y):
        self.tree = DecisionTreeRegressor(criterion=self.criterion)
        self.tree.fit(*self.bootstrap_sample(X, y, self.n_sample))
        print(rank, self.tree)
        return self

    # def predict(self, X):
    #     pred = [self.tree.predict(X.iloc[x].to_frame().T).feature for x in range(len(X))]
    #     y_hat = comm.allreduce(np.array(pred).T, op=MPI_MODE)
    #     return y_hat

    def score(self, X, y):
        assert isinstance(self.tree, DecisionTreeRegressor)
        pred = [self.tree.predict(X.iloc[x].to_frame().T).feature for x in range(len(X))]
        y_hat = comm.allreduce(np.array(pred).T, op=MPI_MODE)
        return mean_squared_error(y, y_hat, squared=False)