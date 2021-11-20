from typing import Union, List

from numpy import ndarray
from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error, r2_score

from firecannon.protocols import Metrics


class SKLearnMetrics(Metrics):
    @staticmethod
    def mse(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def r2(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
        return r2_score(y_true, y_pred)

    @staticmethod
    def accuracy(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
        return precision_score(y_true, y_pred)

    @staticmethod
    def recall(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
        return recall_score(y_true, y_pred)
