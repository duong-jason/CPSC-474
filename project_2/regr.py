#!/usr/bin/env python3


from mpitree.decision_tree import *


if __name__ == "__main__":
    data = {
        'Season': ['winter', 'winter', 'winter', 'spring', 'spring', 'spring', 'summer', 'summer', 'summer', 'autumn', 'autumn', 'autumn'],
        'Work Day': ['false', 'false', 'true', 'false', 'true', 'true', 'false', 'true', 'true', 'false', 'false', 'true'],
        'Rentals': [800, 826, 900, 2100, 4740, 4900, 3000, 5800, 6200, 2910, 2880, 2820],
    }

    df = pd.DataFrame(data)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    dt_regr = DecisionTreeRegressor()
    dt_regr.fit(X, y)

    print(dt_regr)
