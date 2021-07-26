from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

import numpy as np
import pandas as pd
import unittest

from testsuite.services import ParameterizedTestSuite
from testsuite.__tests__.models import LogisticRegression


class ParametrizedTestTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pd.set_option('display.max_columns', None)
        rng = np.random.RandomState(777)
        self.X, self.y = make_classification(n_samples=50, random_state=rng)
        self.classification_stats = ['mean_accuracy', 'std_accuracy', 'mean_recall',
                                     'std_recall', 'mean_precision', 'std_precision']

    def test_with_sklearn_models(self):
        rf = RandomForestClassifier(**{
            "max_features": "sqrt",
            "n_estimators": 200
        })

        knn = KNeighborsClassifier(**{
            "n_neighbors": 5,
            "p": 1
        })

        models = [rf, knn]
        test_settings = {
            'k_folds': 4,
            'float_precision': 3,
            'problem_type': 'classification',
            'stratify': True
        }

        test = ParameterizedTestSuite(test_settings)
        results = test.run(self.X, self.y, models)
        self.assertEqual(list(results.shape), [len(models), len(self.classification_stats)])

    def test_with_personal_model(self):
        lr = LogisticRegression(learning_rate=0.1, fit_intercept=False)
        models = [lr]

        test_settings = {
            'k_folds': 4,
            'float_precision': 3,
            'problem_type': 'classification',
            'stratify': True
        }

        test = ParameterizedTestSuite(test_settings)
        results = test.run(self.X, self.y, models)
        self.assertEqual(list(results.shape), [len(models), len(self.classification_stats)])

    def test_with_list_data(self):
        rf = RandomForestClassifier(**{
            "max_features": "sqrt",
            "n_estimators": 200
        })

        models = [rf]
        test_settings = {
            'k_folds': 4,
            'float_precision': 3,
            'problem_type': 'classification',
            'stratify': True
        }
        self.X = [
            [1, 2, 3],
            [1, 4, 2],
            [2, 1, 4],
            [1, 4, 2],
            [1, 2, 3],
            [1, 3, 2],
            [3, 2, 1],
            [2, 2, 1],
            [1, 3, 1],
            [1, 4, 2]
        ]
        self.y = [0, 0, 1, 1, 1, 0, 1, 0, 1, 0]
        test = ParameterizedTestSuite(test_settings)
        results = test.run(self.X, self.y, models)
        self.assertEqual(list(results.shape), [len(models), len(self.classification_stats)])


if __name__ == '__main__':
    unittest.main()
