import unittest

from quickautoml.entities import Hyperparameter
from quickautoml.estimators import Classifier
from quickautoml.colors import ConsoleColors
from quickautoml.exceptions import InvalidParamException
from quickautoml.services.runners import NaiveModel


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
    my_model = NaiveModel(name="Logistic Regression", estimator=LogisticRegressionStub())
    model_parameters = {
      Hyperparameter(name='learning_rate', data_type='float', min_value=0.01, max_value=1),
    }
    estimator = Classifier(models_settings={
      my_model: model_parameters
    })
    estimator.fit(self.X_train, self.y_train)

  def test_invalid_report_type(self):
    print('Should throw if invalid report type is supplied... ', end='')
    with self.assertRaises(InvalidParamException):
      estimator = Classifier(report_type='invalid')

  def test_valid_report_type(self):
    print('Should be ok if valid report type is supplied...', end='')
    try:
      estimator = Classifier(report_type='plot')
    except Exception as exception:
      self.fail(f'Valid report type supplied throws {exception} unexpectedly')

  def test_invalid_metric(self):
    print('Should throw if invalid metric is supplied... ', end='')
    with self.assertRaises(InvalidParamException):
      estimator = Classifier(metric='invalid')
    
  def test_valid_metric(self):
    print('Should be ok if valid metric is supplied... ', end='')
    try:
      estimator = Classifier(metric='accuracy')
    except Exception as exception:
      self.fail(f'Valid metric supplied throws {exception} unexpectedly')


if __name__ == '__main__':
    unittest.main()
