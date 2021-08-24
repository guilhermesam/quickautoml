from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from os.path import isfile
import unittest
import numpy as np

from firecannon.services import BestParamsTestSuite
from tests.resources import LogisticRegression
import glob
import os


class BestParamsTestTestSuite(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.RandomState(777)
        self.X, self.y = make_classification(n_samples=50, random_state=rng)
        self.test_settings = {
            'verbose': False,
            'k_folds': 4,
            'n_jobs': -1,
            'scoring': 'accuracy'
        }

    def tearDown(self) -> None:
        generated_jsons = glob.glob('*.json')
        for json in generated_jsons:
            os.remove(json)

    def test_with_sklearn_models(self):
        self.model_parameters = {
            KNeighborsClassifier(): {
                'n_neighbors': [3, 5],
                'p': [1, 2],
            }
        }
        self.test_settings.update({'output_path': 'best_knn'})

        self.test_suite = BestParamsTestSuite(self.test_settings)
        self.best_params = self.test_suite.run(self.X, self.y, self.model_parameters)
        self.assertEqual(len(self.best_params.keys()), len(self.model_parameters.keys()))

    def test_with_personal_models(self):
        self.model_parameters = {
            LogisticRegression(): {
                'learning_rate': [0.1, 0.5],
                'fit_intercept': [True, False]
            }
        }
        self.test_settings.update({'output_path': 'best_logistic'})

        self.test_suite = BestParamsTestSuite(self.test_settings)
        self.best_params = self.test_suite.run(self.X, self.y, self.model_parameters)
        self.assertEqual(len(self.best_params.keys()), len(self.model_parameters.keys()))

    def test_json_generation(self):
        self.model_parameters = {
            KNeighborsClassifier(): {
                'n_neighbors': [3, 5],
                'p': [1, 2],
            }
        }
        self.test_settings.update({'output_path': 'test'})

        self.test_suite = BestParamsTestSuite(self.test_settings)
        _ = self.test_suite.run(self.X, self.y, self.model_parameters)

        self.assertTrue(isfile('test.json'))

    def test_with_list_data(self):
        self.model_parameters = {
            KNeighborsClassifier(): {
                'n_neighbors': [3, 5],
                'p': [1, 2],
            }
        }
        self.test_settings.update({'output_path': 'best_knn'})
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
        self.y = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
        self.test_suite = BestParamsTestSuite(self.test_settings)
        self.best_params = self.test_suite.run(self.X, self.y, self.model_parameters)
        self.assertEqual(len(self.best_params.keys()), len(self.model_parameters.keys()))


if __name__ == '__main__':
    unittest.main()
