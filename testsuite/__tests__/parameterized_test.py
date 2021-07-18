from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

import numpy as np
import pandas as pd

from testsuite.services import ParameterizedTestSuite

pd.set_option('display.max_columns', None)

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

test_settings = {
    'k_folds': 4,
    'float_precision': 3,
    'problem_type': 'classification',
    'stratify': True
}

test = ParameterizedTestSuite(test_settings)
results = test.run(X, y, models)

print(results)
