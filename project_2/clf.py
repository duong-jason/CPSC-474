#!/usr/bin/env python3


from mpitree.decision_tree import *


if __name__ == "__main__":
    data = pd.DataFrame({
        'Stream': ["false", "true", "true", "false", "false", "true", "true"],
        'Slope': ["steep", "moderate", "steep", "steep", "flat", "steep", "steep"],
        'Elevation': ["high", "low", "medium", "medium", "high", "highest", "high"],
        'Vegetation': ["chapparal", "riparian", "riparian", "chapparal", "conifer", "conifer", "chapparal"],
    })

    df = pd.DataFrame(data)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X, y)

    print(dt_clf)
