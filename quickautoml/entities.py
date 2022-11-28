from typing import Union, Any, List, Optional, Dict
from abc import ABC
from numpy import ndarray
from dataclasses import dataclass

from quickautoml.protocols import HyperparamsOptimizerDefaults


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


@dataclass
class TrainingConfig:
    report_type: Optional[str]
    hyperparameters_search_space: Dict[NaiveModel, List[Hyperparameter]]
    y_label: str = 'class'
    metric: str = 'recall'

    def __init__(self):
        self.y_label: str = 'class'
        self.metric: str = 'accuracy'
        self.report_type: Optional[str] = None
        self.search_space: dict = {}


class FittedModel(NaiveModel):
    def __init__(self, name: str, cv_score: float, estimator: Any):
        super().__init__(name, estimator)
        self.cv_score: float = cv_score

    def predict(self, x: Union[ndarray, List[list]]):
        return self.estimator.predict(x)


class HyperparamsOptimizer(ABC):
    def __init__(self,
                 scoring: Optional[str]
                 ):
        self.k_folds: int = HyperparamsOptimizerDefaults.k_folds
        self.n_jobs: int = HyperparamsOptimizerDefaults.n_jobs
        self.verbose: int = HyperparamsOptimizerDefaults.verbose
        self.scoring: str = scoring

    def __str__(self) -> str:
        return f'k_folds: {self.k_folds}\n' \
               f'n_jobs: {self.n_jobs}\n' \
               f'verbose: {self.verbose}\n' \
               f'scoring: {self.scoring}'

    def run(self,
            x: Union[ndarray, List[list]],
            y: Union[ndarray, List[list]],
            naive_model: NaiveModel,
            model_settings: List[Hyperparameter]) -> FittedModel:
        pass
