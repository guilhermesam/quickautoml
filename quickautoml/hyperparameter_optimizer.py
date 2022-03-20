from typing import Callable, List, Union

from optuna import Trial, create_study
from optuna.logging import set_verbosity, WARNING
from sklearn.model_selection import GridSearchCV, cross_val_score
from numpy import mean, ndarray

from quickautoml.protocols import VerboseLevels
from quickautoml.entities import NaiveModel, FittedModel, Hyperparameter, HyperparamsOptimizer


class OptunaHyperparamsOptimizer(HyperparamsOptimizer):
  def __init__(self, scoring: str):
    super().__init__(scoring)

  @staticmethod
  def __get_right_suggest_function(trial: Trial, data_type: str) -> Callable:
    return {
      'int': trial.suggest_int,
      'float': trial.suggest_float
    }.get(data_type)

  def run(self,
          x: Union[ndarray, List[list]],
          y: Union[ndarray, List[list]],
          naive_model: NaiveModel,
          model_settings: List[Hyperparameter]) -> FittedModel:
    if self.verbose == VerboseLevels.DISABLED.value:
      set_verbosity(WARNING)

    def objective(trial: Trial) -> float:
      optimizations = {}
      for hyperparameter in model_settings:
        suggest_function = self.__get_right_suggest_function(trial, hyperparameter.data_type)
        optimizations.update({hyperparameter.name: suggest_function(
          name=hyperparameter.name,
          low=hyperparameter.min_value,
          high=hyperparameter.max_value
        )})
      naive_model.estimator = naive_model.estimator.set_params(**optimizations)
      score = cross_val_score(naive_model.estimator, x, y, n_jobs=self.n_jobs, cv=self.k_folds, scoring=self.scoring)
      return float(mean(score))

    study = create_study(direction='maximize',
                         study_name=f'{naive_model.name} Hyperparameter Tunning'
                         )
    study.optimize(objective, n_trials=100)
    best_model = naive_model.estimator.set_params(**study.best_params)
    return FittedModel(
      name=naive_model.name,
      cv_score=study.best_value,
      estimator=best_model
    )


class GridSearchHyperparamsOptimizer(HyperparamsOptimizer):
  def __init__(self, scoring: str):
    super(HyperparamsOptimizer, self).__init__(scoring)

  def run(self,
          x: Union[ndarray, List[list]],
          y: Union[ndarray, List[list]],
          naive_model: NaiveModel,
          model_settings: List[Hyperparameter]) -> FittedModel:
    grid_search = GridSearchCV(estimator=naive_model.estimator,
                               param_grid=model_settings,
                               cv=self.k_folds,
                               verbose=self.verbose,
                               n_jobs=self.n_jobs,
                               scoring=self.scoring)
    grid_search.fit(x, y)

    return FittedModel(
      name=grid_search.best_estimator_.__str__(),
      cv_score=grid_search.best_score_,
      estimator=grid_search.best_estimator_
    )
