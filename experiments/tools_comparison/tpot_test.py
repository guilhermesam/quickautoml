from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from pmlb import fetch_data

X, y = fetch_data('monk1', return_X_y=True)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
                                                    iris.target.astype(np.float64), train_size=0.75, test_size=0.25,
                                                    random_state=42)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X, y)
# print(tpot.score(X_test, y_test))
