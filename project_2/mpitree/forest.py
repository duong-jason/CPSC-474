from mpi4py import MPI
from classifier import *


"""A Distributed Random Forest Algorithm"""


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class RandomForest:
    def __init__(self, n_sample=2, eval="info_gain", criterion={}):
        self.n_sample = n_sample
        self.eval = eval
        self.criterion = criterion
        self.tree = None

    def sub_sample(self, X, n_sample=2):
        """Enforces feature randomness"""
        return np.random.choice(X.columns.to_numpy(), n_sample, replace=False)

    def bootstrap_sample(self, X, y, n_sample, key=True):
        feature_subset = self.sub_sample(X, int(np.log2(len(X.columns))))
        d = pd.concat([X, y], axis=1)
        d = d.sample(n=n_sample, replace=key)
        return d.iloc[:, :-1][feature_subset], d.iloc[:, -1]

    def fit(self, X, y):
        self.tree = DecisionTreeClassifier(eval=self.eval, criterion=self.criterion).fit(*self.bootstrap_sample(X, y, self.n_sample))
        assert isinstance(self.tree, DecisionTreeClassifier)
        print(f"Rank {rank}\n")
        print(self.tree)
        return self

    def predict(self, X):
        return [self.tree.predict(X.iloc[x].to_frame().T) for x in range(len(X))]

    def score(self, X, y):
        buf = []
        buf.append(comm.bcast(self.predict(X), root=rank))

        comm.Barrier()  # Wait for all processes to finish making their predictions

        y_hat = list(map(lambda f: mode(f), np.array(buf).T))
        return accuracy_score(y, y_hat)


def voter(*argv):
    if len(argv) == 1:
        return argv[0]
    for arg in argv:
        if arg is None: continue
        return list(map(lambda f: mode(f), list(zip(voter(arg)))))


MPI_MODE = MPI.Op.Create(voter, commute=True)


if __name__ == '__main__':
    df = pd.read_csv('new_anime.csv')
    df = df.filter(['Rating', 'Studio', 'Source Material', 'Demographic', 'Anime Title'], axis=1)

    bins = 6
    df["Rating"] = pd.cut(df["Rating"], bins, labels=range(bins))

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    comm.Barrier()
    start_time = MPI.Wtime()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    end_time = MPI.Wtime()
    if not rank:
        print(f"Accuracy Score: {accuracy_score(y_test, y_hat)*100:.2f}%")
        print(f"Execution Time: {end_time-start_time:.3f}s")
