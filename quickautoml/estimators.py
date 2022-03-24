from typing import Dict, List

from sklearn.model_selection import train_test_split

from quickautoml.exceptions import InvalidParamException, ModelNotFittedException
from quickautoml.adapters import ModelsSupplier
from quickautoml.reports import BarplotReport, CsvReport, JsonReport
from quickautoml.preprocessors import DataPreprocessor
from quickautoml.entities import NaiveModel, FittedModel, TrainingConfig, Hyperparameter, FeatureEngineer, \
                                 HyperparamsOptimizer
from quickautoml.protocols import ClassifierDefaults


class Classifier:
  def __init__(self,
               data_preprocessor: DataPreprocessor,
               feature_engineer: FeatureEngineer,
               models_supplier: ModelsSupplier,
               hyperparameter_optimizer: HyperparamsOptimizer
               ):
    self.data_preprocessor = data_preprocessor
    self.feature_engineer = feature_engineer
    self.models_supplier = models_supplier
    self.hyperparameter_optimizer = hyperparameter_optimizer
    self.training_config = TrainingConfig()
    self.training_config.search_space = self._default_models_config()
    self.x_test = None
    self.y_test = None

    self.__valid_metrics = ClassifierDefaults.valid_metrics
    self.__valid_report_types = ClassifierDefaults.valid_report_types
    self.best_model = None

    self.__estimator_checks()

  def _default_models_config(self) -> Dict[NaiveModel, List[Hyperparameter]]:
    return {
      NaiveModel(name='KNeighbors Classifier', estimator=self.models_supplier.get_model('knn-c')): [
        Hyperparameter(name='n_neighbors', data_type='int', min_value=3, max_value=7),
        Hyperparameter(name='leaf_size', data_type='int', min_value=15, max_value=60)
      ],
      NaiveModel(name='RandomForest Classifier', estimator=self.models_supplier.get_model('rf-c')): [
        Hyperparameter(name='n_estimators', data_type='int', min_value=10, max_value=60),
        Hyperparameter(name='min_samples_leaf', data_type='int', min_value=2, max_value=64)
      ],
      NaiveModel(name='Adaboost Classifier', estimator=self.models_supplier.get_model('ada-c')): [
        Hyperparameter(name='n_estimators', data_type='int', min_value=10, max_value=100),
        Hyperparameter(name='learning_rate', data_type='float', min_value=0.1, max_value=2)
      ]
    }

  def _check_valid_metric(self) -> None:
    if self.training_config.metric not in self.__valid_metrics:
      raise InvalidParamException(f'Supplied metric is invalid. Choose a value from {self.__valid_metrics}')

  @property
  def cv_score(self):
    if not self.best_model:
      raise ModelNotFittedException("Estimator not fitted yet. Call fit method before")

  def __estimator_checks(self):
    self._check_valid_metric()
    self._check_valid_report_type()

  def _check_valid_report_type(self) -> bool:
    if self.training_config.report_type:
      if self.training_config.report_type in self.__valid_report_types:
        return True
      else:
        raise InvalidParamException(f'Supplied report type is invalid. Choose a value from {self.__valid_report_types}')
    else:
      return False

  def fit(self, data) -> None:
    processed_data = self.data_preprocessor.run(data)
    del data

    feat_engineered_data = self.feature_engineer.run(processed_data)
    del processed_data

    x = feat_engineered_data.drop(self.training_config.y_label, axis=1)
    y = feat_engineered_data[self.training_config.y_label]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.20, random_state=42)

    self.x_test = x_test
    self.y_test = y_test

    scores = {}

    for model, params in self.training_config.search_space.items():
      best_model: FittedModel = self.hyperparameter_optimizer.run(x_train, y_train, model, params)
      scores.update({best_model.estimator: best_model.cv_score})

    self.best_model = self._extract_best_model(scores)
    self.best_model.estimator.fit(x_train, y_train)

    if self._check_valid_report_type():
      self.__make_report_mapper(self.training_config.report_type, scores)

  @staticmethod
  def _extract_best_model(scores):
    best_model = max(scores, key=scores.get)
    return FittedModel(
      name=best_model.__str__(),
      cv_score=max(scores.values()),
      estimator=best_model
    )

  def predict(self):
    return self.best_model.predict(self.x_test)

  @staticmethod
  def __make_report_mapper(report_type: str, scores: dict):
    report_types = {
      'plot': BarplotReport(),
      'csv': CsvReport(),
      'json': JsonReport()
    }
    report_types.get(report_type).make_report(scores)
