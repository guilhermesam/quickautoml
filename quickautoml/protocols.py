from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List
from dataclasses import dataclass

from quickautoml.entities import NaiveModel, Hyperparameter


class VerboseLevels(Enum):
    DISABLED = 0
    ENABLED = 1


@dataclass
class ClassifierDefaults:
    valid_metrics = ['accuracy', 'precision', 'recall']
    valid_report_types = ['plot', 'csv', 'json']
