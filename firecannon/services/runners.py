from abc import abstractmethod

from firecannon.exceptions import InvalidParamException
from firecannon.services.hyperparams_tunners import OptunaHyperparamsTunner
from firecannon.adapters.models_adapters import SKLearnModelsSupplier, ModelsSupplier
from firecannon.reports import BarplotReport, CsvReport, JsonReport
from firecannon.utils import check_shape_compatibility
from firecannon.preprocessors import DataPreprocessor
from firecannon.entities import NaiveModel, FittedModel, Hyperparameter


class BaseModel:
  def __init__(self,
               metric: str,
               report_type: str = None,
               models_settings: str = None,
               models_supplier: ModelsSupplier = SKLearnModelsSupplier()):
    self.best_model: FittedModel
    self.metric = metric
    self.report_type = report_type
    self._models_supplier = models_supplier
    self.__valid_report_types = [
      'plot', 'csv', 'json'
    ]
    self.__estimator_checks()
    self.model_settings = self._default_models_config() if not models_settings else models_settings
    self.preprocessor = DataPreprocessor()

  @property
  def cv_score(self):
    return self.best_model.cv_score

  def __estimator_checks(self):
    self._check_valid_metric()
    self._check_valid_report_type()

  def _check_valid_metric(self):
    """Deve ser implementado na subclasse"""
    pass

  def _check_valid_report_type(self) -> bool:
    if self.report_type:
      if self.report_type in self.__valid_report_types:
        return True
      else:
        raise InvalidParamException(f'Supplied report type is invalid. Choose a value from {self.__valid_report_types}')
    else:
      return False

  @abstractmethod
  def _default_models_config(self):
    raise NotImplementedError('Implement default models to test')

  def fit(self, X, y) -> None:
    check_shape_compatibility(X, y)
    X = self.preprocessor.run(X)

    hyperparams_tunner = OptunaHyperparamsTunner(scoring=self.metric)

    scores = {}

    for model, params in self.model_settings.items():
      best_model: FittedModel = hyperparams_tunner.run(X, y, model, params)
      scores.update({best_model.estimator: best_model.cv_score})

    self.best_model = self._extract_best_model(scores)
    self.best_model.estimator.fit(X, y)

    if self._check_valid_report_type():
      self.make_report_mapper(self.report_type, scores)

  @staticmethod
  def _extract_best_model(scores):
    best_model = max(scores, key=scores.get)
    return FittedModel(
      name=best_model.__str__(),
      cv_score=max(scores.values()),
      estimator=best_model
    )

  def predict(self, X):
    return self.best_model.predict(X)

  @staticmethod
  def make_report_mapper(report_type: str, scores: dict):
    report_types = {
      'plot': BarplotReport(),
      'csv': CsvReport(),
      'json': JsonReport()
    }
    report_types.get(report_type).make_report(scores)


class Regressor(BaseModel):
  def __init__(self,
               metric: str = 'r2',
               report_type: str = None,
               models_settings: dict = None
               ) -> None:
    super().__init__(metric, models_settings, report_type)

  def _default_models_config(self):
    self.model_settings = {
      self._models_supplier.get_model('rf-r'): {
        'n_estimators': [50, 100, 150],
        'criterion': ['mse', 'mae'],
        'max_features': ['auto', 'log2', 'sqrt'],
      },
      self._models_supplier.get_model('lasso'): {
        'alpha': [1.0, 1.5, 2.0],
        'fit_intercept': [True, False],
        'normalize': [True, False]
      },
      self._models_supplier.get_model('en'): {
        'alpha': [1.0, 1.5, 2.0],
        'fit_intercept': [True, False],
        'normalize': [True, False]
      }
    }
 

class Classifier(BaseModel):
  def __init__(self,
               metric: str = 'accuracy',
               report_type: str = None,
               models_settings: dict = None
               ) -> None:
    self.__valid_metrics = ['accuracy', 'precision', 'recall']
    self.report_type = report_type
    super().__init__(metric, report_type, models_settings)

  def _check_valid_metric(self):
    if self.metric not in self.__valid_metrics:
      raise InvalidParamException(f'Supplied report type is invalid. Choose a value from {self.__valid_metrics}')

  def _default_models_config(self):
    return {
      NaiveModel(name='KNeighbors Classifier', estimator=self._models_supplier.get_model('knn-c')): [
        Hyperparameter(name='n_neighbors', data_type='int', min_value=3, max_value=7),
        Hyperparameter(name='leaf_size', data_type='int', min_value=15, max_value=60)
      ],
      NaiveModel(name='RandomForest Classifier', estimator=self._models_supplier.get_model('rf-c')): [
        Hyperparameter(name='n_estimators', data_type='int', min_value=10, max_value=60),
        Hyperparameter(name='min_samples_leaf', data_type='int', min_value=2, max_value=64)
      ],
      NaiveModel(name='Adaboost Classifier', estimator=self._models_supplier.get_model('ada-c')): [
        Hyperparameter(name='n_estimators', data_type='int', min_value=10, max_value=100),
        Hyperparameter(name='learning_rate', data_type='float', min_value=0.1, max_value=2)
      ]
    }
