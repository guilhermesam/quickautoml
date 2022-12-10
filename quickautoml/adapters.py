from abc import ABC, abstractmethod
from typing import Union, List

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from numpy import ndarray
from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error, r2_score

from quickautoml.protocols import Metrics


class ModelsSupplier(ABC):
    def __init__(self) -> None:
        self._models_codes = None

    @abstractmethod
    def get_model(self, model_name: str):
        """
    Deve retornar uma instância de um modelo, de acordo com o nome do mesmo. Possíveis nomes:
    Classificadores:
    knn-c: KNearestNeighbors Classifier
    rf-c: RandomForest Classifier
    ada-c: AdaBoost Classifier

    knn-r: KNearestNeighbors Regressor
    rf-r: RandomForest Regressor
    lasso: Lasso Regressor
    en: ElasticNet Regressor
    """
        pass


class SKLearnModelsSupplier(ModelsSupplier):
    def __init__(self) -> None:
        super().__init__()
        self._models_codes = {
            'knn-c': KNeighborsClassifier(),
            'rf-c': RandomForestClassifier(),
            'ada-c': AdaBoostClassifier(),
            'svc': SVC(),
            'dt': DecisionTreeClassifier(),
            'rf-r': RandomForestRegressor(),
            'lasso': Lasso(),
            'en': ElasticNet()
        }

    def get_model(self, model_name: str) -> object:
        return self._models_codes.get(model_name)


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
