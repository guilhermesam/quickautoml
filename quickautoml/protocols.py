from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union
from dataclasses import dataclass

from numpy import ndarray


class VerboseLevels(Enum):
    DISABLED = 0
    ENABLED = 1


@dataclass
class ClassifierDefaults:
    valid_metrics = ['accuracy', 'precision', 'recall']
    valid_report_types = ['plot', 'csv', 'json']


@dataclass()
class HyperparamsOptimizerDefaults:
    k_folds: int = 5
    n_jobs: int = -1
    verbose: int = 0
    scoring: str = 'accuracy'


class Metrics(ABC):
    @staticmethod
    @abstractmethod
    def mse(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
        pass

    @staticmethod
    @abstractmethod
    def r2(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
        pass

    @staticmethod
    @abstractmethod
    def accuracy(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
        pass

    @staticmethod
    @abstractmethod
    def precision(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
        pass

    @staticmethod
    @abstractmethod
    def recall(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
        pass
