from typing import Callable, List, Union, Dict, Any, Iterable

from optuna import Trial, create_study
from optuna.pruners import ThresholdPruner
from optuna.logging import set_verbosity, WARNING
from optuna.exceptions import TrialPruned

from sklearn.model_selection import cross_val_score
from numpy import mean, ndarray

from quickautoml.classifiers import Estimators
from quickautoml.entities import NaiveModel, FittedModel, Hyperparameter


class HyperparamsOptimizer:
    def __init__(self,
                 metric: str,
                 search_space: Dict[NaiveModel, List[Hyperparameter]] = None,
                 n_jobs: int = None,
                 k_folds: int = 5,
                 stopping_criteria: float = 0.9
                 ):
        self.metric: str = metric
        self._estimators_supplier = Estimators()
        self.search_space: Dict[NaiveModel, List[Hyperparameter]] = search_space or self._default_search_space()
        self.n_jobs: int = n_jobs
        self.k_folds: int = k_folds
        self.stopping_criteria = stopping_criteria

    def _default_search_space(self) -> Dict[NaiveModel, List[Hyperparameter]]:
        return {
            NaiveModel(name='KNeighbors Classifier', estimator=self._estimators_supplier.get_classifier('knn-c')): [
                Hyperparameter(name='n_neighbors', data_type='int', min_value=3, max_value=7),
                Hyperparameter(name='leaf_size', data_type='int', min_value=15, max_value=60)
            ],
            NaiveModel(name='RandomForest Classifier', estimator=self._estimators_supplier.get_classifier('rf-c')): [
                Hyperparameter(name='n_estimators', data_type='int', min_value=10, max_value=60),
                Hyperparameter(name='min_samples_leaf', data_type='int', min_value=2, max_value=64)
            ],
            NaiveModel(name='Adaboost Classifier', estimator=self._estimators_supplier.get_classifier('ada-c')): [
                Hyperparameter(name='n_estimators', data_type='int', min_value=10, max_value=100),
                Hyperparameter(name='learning_rate', data_type='float', min_value=0.1, max_value=2)
            ]
        }

    @staticmethod
    def __get_suggest_function(trial: Trial, data_type: str) -> Callable:
        return {
            'int': trial.suggest_int,
            'float': trial.suggest_float
        }.get(data_type)

    @staticmethod
    def __get_best_model(candidate_models: Iterable) -> FittedModel:
        return max(candidate_models)

    def run(self, train_data: Any, labels: Any) -> FittedModel:
        fitted_models: List[FittedModel] = []
        for model, hyperparameters in self.search_space.items():
            fitted_model = self.__run_study(train_data, labels, model, hyperparameters)
            fitted_models.append(fitted_model)

        return self.__get_best_model(fitted_models)

    def __run_study(self,
                    train_data: Union[ndarray, List[list]],
                    labels: Union[ndarray, List[list]],
                    naive_model: NaiveModel,
                    hyperparameters: List[Hyperparameter]) -> FittedModel:
        def callback(study, trial):
            if trial.value > 0.95:
                raise TrialPruned

        def objective(trial: Trial) -> float:
            optimizations = {}
            for hyperparameter in hyperparameters:
                suggest_function = self.__get_suggest_function(trial, hyperparameter.data_type)
                optimizations.update({hyperparameter.name: suggest_function(
                    name=hyperparameter.name,
                    low=hyperparameter.min_value,
                    high=hyperparameter.max_value
                )})
            naive_model.estimator = naive_model.estimator.set_params(**optimizations)
            score = cross_val_score(naive_model.estimator,
                                    train_data,
                                    labels,
                                    n_jobs=self.n_jobs,
                                    cv=self.k_folds,
                                    scoring=self.metric)

            return float(mean(score))

        hyperparam_optimization_study = create_study(direction='maximize',
                                                     study_name=f'{naive_model.name} Hyperparameter Tunning',
                                                     pruner=ThresholdPruner(upper=self.stopping_criteria)
                                                     )
        try:
            hyperparam_optimization_study.optimize(objective, callbacks=[callback])
        except TrialPruned:
            best_metric = hyperparam_optimization_study.best_value
            print(f"Hyperparameter optimization step has been pruned! - Metric: {best_metric}")

        fitted_model = FittedModel(
            name=naive_model.name,
            estimator=naive_model.estimator.set_params(**hyperparam_optimization_study.best_params),
            cv_score=hyperparam_optimization_study.best_value
        )

        return fitted_model
