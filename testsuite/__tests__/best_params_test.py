from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

from testsuite.services import *

import unittest
from os.path import isfile


class BestParamsTestTestSuite(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.RandomState(777)
        self.X, self.y = make_classification(n_samples=600, random_state=rng)
        self.test_settings = {
            'verbose': False,
            'output_path': None,
            'k_folds': 4,
            'n_jobs': -1
        }

    def test_with_sklearn_models(self):
        model_parameters = {
            KNeighborsClassifier(): {
                'n_neighbors': [3, 5],
                'p': [1, 2],
            }
        }

        test_suite = BestParamsTestSuite(self.test_settings)
        best_params = test_suite.run(self.X, self.y, model_parameters)
        self.assertEqual(len(best_params.keys()), len(model_parameters.keys()))

    def test_json_generation(self):
        model_parameters = {
            KNeighborsClassifier(): {
                'n_neighbors': [3, 5],
                'p': [1, 2],
            }
        }
        self.test_settings.update({'output_path': 'best_models'})

        test_suite = BestParamsTestSuite(self.test_settings)
        _ = test_suite.run(self.X, self.y, model_parameters)

        self.assertTrue(isfile('best_models.json'))
