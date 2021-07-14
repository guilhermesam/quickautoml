from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, recall_score, precision_score

import numpy as np

from testsuite.utils import \
    check_shape_compatibility, \
    write_json, \
    convert_to_np_array


class BestParamsTestSuite:
    """
    Class to find best hyperparameters for a list of models
    """

    def __init__(self, verbose: bool = False, output_path: str = None, k_folds: int = 4, n_jobs: int = -1):
        """
        Parameters
        ----------
        verbose: if True, shows in console the best params for each model
        output_path: if provided, saves the best params for each model in a json
        k_folds: number of folds in cross validation step
        n_jobs: number of jobs running in paralel
        """
        self.verbose = verbose
        self.output_path = output_path
        self.k_folds = k_folds
        self.n_jobs = n_jobs

    def run(self, x: any, y: any, model_parameters: dict):
        """
        x: any implementation of 2D matrix with features for training
        y: any implementation of 2D matrix with labels
        model_parameters: dict, which keys are instantiated models and values are lists with hyperparameter
        """
        if not check_shape_compatibility(x, y):
            raise ValueError('X e y possuem valores inconsistentes de amostras!')

        best_models = {}

        for model in model_parameters.keys():
            grid_search = GridSearchCV(estimator=model,
                                       param_grid=model_parameters.get(model),
                                       cv=self.k_folds,
                                       n_jobs=self.n_jobs)
            grid_search.fit(x, y)
            best_params = grid_search.best_params_

            best_models.update({model.__class__.__name__: best_params})

            if self.verbose:
                print(f'Best params for {model.__class__.__name__}: {best_params}')

        if self.output_path:
            write_json(data=best_models, filepath=self.output_path)

        return best_models


class ParameterizedTestSuite:
    """
    Class to test models with tuned hyperparameters
    """

    def __init__(self, stratify: bool = False, k_folds: int = 4, float_precision: int = 3):
        """
        Parameters
        ----------
        stratify: if True, get stratified samples for each class in the features vector. Recommended for
            classification problems
        """
        self.stratify = stratify
        self.k_folds = k_folds
        self.float_precision = float_precision

    def __make_folds(self):
        if self.stratify:
            return StratifiedKFold(n_splits=self.k_folds)
        else:
            return KFold(n_splits=self.k_folds)

    def run(self, x: any, y: any, models: list) -> DataFrame:
        """
        Parameters
        ----------
        x: any implementation of 2D matrix with features for training
        y: any implementation of 2D matrix with labels
        models: list of instantiated models
        -------

        """
        if not check_shape_compatibility(x, y):
            raise ValueError('X e y possuem valores inconsistentes de amostras!')

        x = convert_to_np_array(x)
        y = convert_to_np_array(y)

        scores_df = DataFrame()
        kfold = self.__make_folds()

        for model in models:
            accuracy = []
            recall = []
            precision = []

            for train, test in kfold.split(x, y):
                model.fit(x[train], y[train])
                predictions = model.predict(x[test])
                accuracy.append(accuracy_score(y[test], predictions))
                recall.append(recall_score(y[test], predictions))
                precision.append(precision_score(y[test], predictions))

            score_series = Series({
                'mean_accuracy': np.round(np.mean(accuracy), self.float_precision),
                'std_accuracy': np.round(np.std(accuracy), self.float_precision),
                'mean_recall': np.round(np.mean(recall), self.float_precision),
                'std_recall': np.round(np.std(recall), self.float_precision),
                'mean_precision': np.round(np.mean(precision), self.float_precision),
                'std_precision': np.round(np.std(precision), self.float_precision)
            }, name=model.__class__.__name__)

            scores_df = scores_df.append(score_series)

        return scores_df
