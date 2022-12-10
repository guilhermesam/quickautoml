from typing import Callable, Union, List

from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from numpy import ndarray


def get_metric_as_callable(name: str) -> Callable:
    return {
        'accuracy': accuracy_wrapper,
        'precision': precision_wrapper,
        'recall': recall_wrapper,
        'f1_score': f1_wrapper,
        'specificity': specificity_wrapper,
        'sensitivity': sensitivity_wrapper,
        'roc_curve': roc_curve_wrapper,
        'auc_score': auc_score_wrapper
    }.get(name)


def accuracy_wrapper(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]) -> float:
    return accuracy_score(y_true, y_pred)


def precision_wrapper(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
    return precision_score(y_true, y_pred)


def recall_wrapper(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
    return recall_score(y_true, y_pred)


def f1_wrapper(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
    return f1_score(y_true, y_pred)


def specificity_wrapper(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def sensitivity_wrapper(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tn + fn)


def roc_curve_wrapper(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
    return roc_curve(y_true, y_pred)


def auc_score_wrapper(y_true: Union[ndarray, List[float]], y_pred: Union[ndarray, List[float]]):
    return roc_auc_score(y_true, y_pred)
