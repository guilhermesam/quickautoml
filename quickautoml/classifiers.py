from abc import ABC, abstractmethod
from typing import Union, List

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Estimators:
    def __init__(self) -> None:
        super().__init__()
        self._estimator_codes = {
            'knn-c': KNeighborsClassifier(),
            'rf-c': RandomForestClassifier(),
            'ada-c': AdaBoostClassifier(),
            'svc': SVC(),
            'dt': DecisionTreeClassifier(),
            'rf-r': RandomForestRegressor(),
            'lasso': Lasso(),
            'en': ElasticNet()
        }

    def get_classifier(self, model_name: str) -> object:
        return self._estimator_codes.get(model_name)
