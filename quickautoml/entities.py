from typing import Union, Any, List, Optional, Dict
from numpy import ndarray
from dataclasses import dataclass


class NaiveModel:
    def __init__(self, name: str, estimator: Any):
        self.name: str = name
        self.estimator: Any = estimator

    def __str__(self) -> str:
        return self.estimator.__str__()


@dataclass
class Hyperparameter:
    name: str
    data_type: str
    min_value: float
    max_value: float


class FittedModel(NaiveModel):
    def __init__(self, name: str, cv_score: float, estimator: Any):
        super().__init__(name, estimator)
        self.cv_score: float = cv_score

    def __gt__(self, other):
        return self.cv_score > other.cv_score

    def predict(self, x: Union[ndarray, List[list]]):
        return self.estimator.predict(x)
