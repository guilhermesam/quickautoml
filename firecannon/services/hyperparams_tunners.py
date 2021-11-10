from abc import ABC
from typing import Callable, List
from os.path import join
from pathlib import Path

from optuna import Trial, create_study
from optuna.logging import set_verbosity, WARNING
from sklearn.model_selection import GridSearchCV, cross_val_score
from numpy import mean

from firecannon.protocols import VerboseLevels
from firecannon.entities import NaiveModel, FittedModel, Hyperparameter
from firecannon.utils import load_json


class HyperparamsTunnerBase(ABC):
  def __init__(self,
               scoring: str
               ):
    __dir = Path(__file__).parent.resolve()
    default_config: dict = load_json(join(__dir, '../default_config.json'))
    self.scoring = scoring
    self.k_folds = default_config['k_folds']
    self.n_jobs = default_config['n_jobs']
    self.verbose = default_config['verbose']

  def __str__(self):
    return f'k_folds: {self.k_folds}\n' \
           f'n_jobs: {self.n_jobs}\n' \
           f'verbose: {self.verbose}\n' \
           f'scoring: {self.scoring}'

  def run(self, X: any, y: any, model: any, model_settings: dict):
    pass


class OptunaHyperparamsTunner(HyperparamsTunnerBase):
  def __init__(self, scoring: str):
    super().__init__(scoring)

  @staticmethod
  def __get_right_suggest_function(trial: Trial, data_type: str) -> Callable:
    return {
      'int': trial.suggest_int,
      'float': trial.suggest_float
    }.get(data_type)

  def run(self, X: any, y: any, naive_model: NaiveModel, model_settings: List[Hyperparameter]):
    if self.verbose == VerboseLevels.DISABLED.value:
      set_verbosity(WARNING)

    def objective(trial):
      optimizations = {}
      for hyperparameter in model_settings:
        suggest_function = self.__get_right_suggest_function(trial, hyperparameter.data_type)
        optimizations.update({hyperparameter.name: suggest_function(
          name=hyperparameter.name,
          low=hyperparameter.min_value,
          high=hyperparameter.max_value
        )})
      naive_model.estimator = naive_model.estimator.set_params(**optimizations)
      score = cross_val_score(naive_model.estimator, X, y, n_jobs=self.n_jobs, cv=self.k_folds, scoring=self.scoring)
      return mean(score)

    study = create_study(direction='maximize',
                         study_name=f'{naive_model.name} Hyperparameter Tunning'
                         )
    study.optimize(objective, n_trials=50)
    best_model = naive_model.estimator.set_params(**study.best_params)
    return FittedModel(
      name=naive_model.name,
      cv_score=study.best_value,
      estimator=best_model
    )


class GridSearchHyperparamsTunner(HyperparamsTunnerBase):
  def __init__(self, scoring: str):
    super(HyperparamsTunnerBase, self).__init__(scoring)

  def run(self, X: any, y: any, model: any, model_settings: dict):
    grid_search = GridSearchCV(estimator=model,
                               param_grid=model_settings,
                               cv=self.k_folds,
                               verbose=self.verbose,
                               n_jobs=self.n_jobs,
                               scoring=self.scoring or model.score)
    grid_search.fit(X, y)

    return FittedModel(
      name=grid_search.best_estimator_.__str__(),
      cv_score=grid_search.best_score_,
      estimator=grid_search.best_estimator_
    )