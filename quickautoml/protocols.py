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


@dataclass
class HyperparamsOptimizerDefaults:
    k_folds: int = 5
    n_jobs: int = -1
    verbose: int = 1
    scoring: str = 'recall'
