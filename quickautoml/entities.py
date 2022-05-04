from typing import Union, Any, List, Optional, Dict
from abc import ABC
from numpy import ndarray
from pandas import DataFrame

from quickautoml.protocols import HyperparamsOptimizerDefaults


class TrainingConfig:
  def __init__(self):
    self.y_label: str = 'class'
    self.metric: str = 'accuracy'
    self.report_type: Optional[str] = None
    self.search_space: dict = {}


class NaiveModel:
  def __init__(self, name: str, estimator: Any):
    self.name: str = name
    self.estimator: Any = estimator

  def __str__(self) -> str:
    return self.estimator.__str__()


class Hyperparameter:
  def __init__(self,
               name: str,
               data_type: str,
               min_value: float,
               max_value: float):
    self.name: str = name
    self.data_type: str = data_type
    self.min_value: float = min_value
    self.max_value: float = max_value


class FittedModel(NaiveModel):
  def __init__(self, name: str, cv_score: float, estimator: Any):
    super().__init__(name, estimator)
    self.cv_score: float = cv_score

  def predict(self, x: Union[ndarray, List[list]]):
    return self.estimator.predict(x)


class DataPreprocessor:
  def __init__(self):
    self.matrix = None

  def __convert_to_dataframe(self):
    ...

  def __convert_df_to_np_array(self):
    ...

  def __remove_null_values(self):
    ...

  def __remove_duplicates(self):
    ...

  def __collect(self):
    ...

  def run(self, matrix: Union[DataFrame, ndarray]):
    ...


class FeatureEngineer:
  def __init__(self):
    self.matrix = None

  def __remove_unbalanced_columns(self):
    ...

  def __count_used_permissions(self):
    ...

  def __remove_columns_with_unique_values(self):
    ...

  def __collect(self):
    ...

  def run(self, matrix: Union[DataFrame, ndarray]):
    ...


class HyperparamsOptimizer(ABC):
  def __init__(self,
               scoring: Optional[str]
               ):
    self.k_folds: int = HyperparamsOptimizerDefaults.k_folds
    self.n_jobs: int = HyperparamsOptimizerDefaults.n_jobs
    self.verbose: int = HyperparamsOptimizerDefaults.verbose
    self.scoring: str = scoring

  def __str__(self) -> str:
    return f'k_folds: {self.k_folds}\n' \
           f'n_jobs: {self.n_jobs}\n' \
           f'verbose: {self.verbose}\n' \
           f'scoring: {self.scoring}'

  def run(self,
          x: Union[ndarray, List[list]],
          y: Union[ndarray, List[list]],
          naive_model: NaiveModel,
          model_settings: List[Hyperparameter]) -> FittedModel:
    pass
