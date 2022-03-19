from abc import abstractmethod
from typing import Dict, List

from quickautoml.exceptions import InvalidParamException, ModelNotFittedException
from quickautoml.adapters import ModelsSupplier
from quickautoml.reports import BarplotReport, CsvReport, JsonReport
from quickautoml.preprocessors import DataPreprocessor
from quickautoml.entities import NaiveModel, FittedModel, TrainingConfig, Hyperparameter
from quickautoml.services import OptunaHyperparamsOptimizer, HyperparamsOptimizer
from quickautoml.protocols import ClassifierDefaults

"""
DataPreprocessor()
FeatureEngineer()
ModelsSupplier()
HyperparameterOptimizer()
"""


class BaseModel:
  def __init__(self,
               model_config: TrainingConfig,
               hyperparameter_optimizer: HyperparamsOptimizer,
               models_supplier: ModelsSupplier):
    self.best_model = None
    self.metric = model_config.metric
    self.report_type = model_config.report_type
    self.hyperparameter_optimizer = hyperparameter_optimizer
    self._models_supplier = models_supplier
    self.__valid_report_types = [
      'plot', 'csv', 'json'
    ]
    self.__estimator_checks()
    self.model_settings = self._default_models_config()
    self.preprocessor = DataPreprocessor()

  @property
  def cv_score(self):
    if not self.best_model:
      raise ModelNotFittedException("Estimator not fitted yet. Call fit method before")

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
  def _default_models_config(self) -> dict:
    raise NotImplementedError('Implement default models to test')

  def fit(self, x, y) -> None:
    # check_shape_compatibility(X, y)
    # X = self.preprocessor.run(X)

    hyperparams_tunner = OptunaHyperparamsOptimizer(scoring=self.metric)

    scores = {}

    for model, params in self.model_settings.items():
      best_model: FittedModel = hyperparams_tunner.run(x, y, model, params)
      scores.update({best_model.estimator: best_model.cv_score})

    self.best_model = self._extract_best_model(scores)
    self.best_model.estimator.fit(x, y)

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

  def predict(self, x):
    return self.best_model.predict(x)

  @staticmethod
  def make_report_mapper(report_type: str, scores: dict):
    report_types = {
      'plot': BarplotReport(),
      'csv': CsvReport(),
      'json': JsonReport()
    }
    report_types.get(report_type).make_report(scores)


class Classifier(BaseModel):
  def __init__(self,
               model_config: TrainingConfig,
               hyperparameter_optimizer: HyperparamsOptimizer,
               models_supplier: ModelsSupplier
               ):
    self.__valid_metrics = ClassifierDefaults.valid_metrics
    super().__init__(model_config, hyperparameter_optimizer, models_supplier)

  def _check_valid_metric(self) -> None:
    if self.metric not in self.__valid_metrics:
      raise InvalidParamException(f'Supplied report type is invalid. Choose a value from {self.__valid_metrics}')

  def _default_models_config(self) -> Dict[NaiveModel, List[Hyperparameter]]:
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
