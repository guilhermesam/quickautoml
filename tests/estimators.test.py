import unittest
from firecannon.estimators import Classifier
from firecannon.colors import ConsoleColors
from firecannon.exceptions import InvalidParamException

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from tests.resources import LogisticRegressionStub


class EstimatorsTestCase(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    print(f'{ConsoleColors.BOLD}{ConsoleColors.OKCYAN}Estimators Test Suite{ConsoleColors.END_COLORS}')

  def setUp(self) -> None:
    self.X, self.y = make_classification(n_samples=100, n_features=5, n_classes=2)
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25)

  def tearDown(self) -> None:
    print(f'{ConsoleColors.OKGREEN}OK{ConsoleColors.END_COLORS}')

  def test_personal_models(self):
    print('Should be ok with valid personal models... ', end='')
    my_model = LogisticRegressionStub()
    model_parameters = {
      'learning_rate': [0.01, 0.1, 1],
      'fit_intercept': [True, False],
    }
    try:
      estimator = Classifier(models_settings={
        my_model: model_parameters
      })
      estimator.fit(self.X_train, self.y_train)
    except Exception as exception:
      self.fail(f'Fitted estimator with personal model raises {exception} unexpectedly')

  def test_invalid_report_type(self):
    print('Should throw if invalid report type is supplied... ', end='')
    with self.assertRaises(InvalidParamException):
      estimator = Classifier(report_type='invalid')


if __name__ == '__main__':
    unittest.main()
