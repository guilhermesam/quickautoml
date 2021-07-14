from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from test_suite import *

import numpy as np
import pandas as pd

rng = np.random.RandomState(777)
X, y = make_classification(n_samples=600, random_state=rng)

X = pd.DataFrame(X)

models = [
  RandomForestClassifier(**{
    "max_features": "sqrt",
    "n_estimators": 200
  }),
  KNeighborsClassifier(**{
    "n_neighbors": 5,
    "p": 1
  })
]

test_parameters = {
    'n_splits': 3,
    'float_precision': 3
}

test = ParameterizedTestSuite(stratify=True)
results = test.run(X, y, models, test_parameters)

print(results)