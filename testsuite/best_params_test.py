import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from test_suite import *

rng = np.random.RandomState(777)

X, y = make_classification(n_samples=600, random_state=rng)

model_parameters = {
    KNeighborsClassifier(): {
        'n_neighbors': [3, 5],
        'p': [1, 2],
    },
    RandomForestClassifier(): {
        'n_estimators': [100, 200],
        'max_features': ['auto', 'sqrt'],
    }
}

test_parameters = {
    'cv': 3,
}

test_suite = BestParamsTestSuite(output_file=True)
best_params = test_suite.run(X, y, model_parameters, test_parameters)

print(best_params)