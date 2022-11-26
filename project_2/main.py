from mpi4py import MPI
from pickle import load
from sklearn.model_selection import train_test_split


from mpitree.random_forest import *


if __name__ == '__main__':
    df = load(open('df_anime_x.p', 'rb'))
    df = df[['Source Material', 'Demographic', 'Rating']]

    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    comm.Barrier()
    start_time = MPI.Wtime()

    dt_regr = RandomForest(n_sample=len(X_train), criterion={'max_depth': 1})
    dt_regr.fit(X_train, y_train)

    test = pd.DataFrame({"Source Material": ["Manga"], "Demographic": ["Kids"]})
    score = dt_regr.predict(test)
    # score = dt_regr.score(X_test, y_test)

    end_time = MPI.Wtime()
    if not rank:
        # print(f"MSE Score: {score:.2f}")
        print("Score:", score)
        print(f"Execution Time: {end_time-start_time:.3f}s")
