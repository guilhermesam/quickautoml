from typing import Union, List

from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, mean_squared_error, r2_score
from numpy import ndarray


class Metrics:
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

    @staticmethod
    def f1_score(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
        return f1_score(y_true, y_pred)
