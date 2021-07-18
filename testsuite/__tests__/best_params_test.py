from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from testsuite.services import *

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

test_settings = {
    'verbose': False,
    'output_path': 'best_models.json',
    'k_folds': 4,
    'n_jobs': -1
}

test_suite = BestParamsTestSuite(test_settings)
best_params = test_suite.run(X, y, model_parameters)
print(test_suite)
