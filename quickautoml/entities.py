from typing import Union, Any, List, Optional, Dict
from abc import ABC
from numpy import ndarray
from pandas import DataFrame


class TrainingConfig:
  def __init__(self):
    self.metric: str = 'accuracy'
    self.report_type: Optional[str] = None
    self.search_space: dict = {}


class NaiveModel:
  def __init__(self, name: str, estimator: Any):
    self.name: str = name
    self.estimator: Any = estimator

  def __str__(self) -> str:
    return self.estimator.__str__()


class FittedModel(NaiveModel):
  def __init__(self, name: str, cv_score: float, estimator: Any):
    super().__init__(name, estimator)
    self.cv_score: float = cv_score

  def predict(self, x: Union[ndarray, List[list]]):
    return self.estimator.predict(x)


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


class Pipeline(ABC):
  def execute(self) -> DataFrame:
    ...
