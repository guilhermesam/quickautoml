from abc import ABC, abstractmethod
from enum import Enum


class VerboseLevels(Enum):
    DISABLED = 0
    ENABLED = 1


class Metrics(ABC):
    @staticmethod
    @abstractmethod
    def mse(y_true, y_pred):
        pass

    @staticmethod
    @abstractmethod
    def r2(y_true, y_pred):
        pass

    @staticmethod
    @abstractmethod
    def accuracy(y_true, y_pred):
        pass

    @staticmethod
    @abstractmethod
    def precision(y_true, y_pred):
        pass

    @staticmethod
    @abstractmethod
    def recall(self, y_true, y_pred):
        pass
