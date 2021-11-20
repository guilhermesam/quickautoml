from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Union

from numpy import ndarray


class VerboseLevels(Enum):
    DISABLED = 0
    ENABLED = 1


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
