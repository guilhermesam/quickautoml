from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_absolute_error, r2_score

import numpy as np

from testsuite.utils import check_shape_compatibility, write_json, convert_to_np_array


class BestParamsTestSuite:
    """
    Class to find best hyperparameters for a list of models
    """
    def __init__(self, test_settings: dict):
        self.k_folds = test_settings.get('k_folds') or 4
        self.n_jobs = test_settings.get('n_jobs') or -1
        self.verbose = test_settings.get('verbose') or False
        self.output_path = test_settings.get('output_path') or 'NO_PATH'
        self.scoring = test_settings.get('scoring')

    def __str__(self):
        return f'k_folds: {self.k_folds}\n' \
               f'n_jobs: {self.n_jobs}\n' \
               f'verbose: {self.verbose}\n' \
               f'output_path: {self.output_path}'

    def run(self, x: any, y: any, model_parameters: dict) -> dict:
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
                                       n_jobs=self.n_jobs,
                                       scoring=self.scoring or model.score)
            grid_search.fit(x, y)
            best_params = grid_search.best_params_

            best_models.update({model.__class__.__name__: best_params})

            if self.verbose:
                print(f'Best params for {model.__class__.__name__}: {best_params}')

        if self.output_path != 'NO_PATH':
            write_json(data=best_models, filepath=self.output_path)

        return best_models


class ParameterizedTestSuite:
    """
    Class to test models with tuned hyperparameters
    """

    def __init__(self, test_settings: dict):
        self.problem_type = test_settings.get('problem_type')
        self.k_folds = test_settings.get('k_folds') or 4
        self.stratify = test_settings.get('stratify') or False
        self.float_precision = test_settings.get('float_precision') or 3

    def __str__(self):
        return f'k_folds: {self.k_folds}\n' \
               f'problem_type: {self.problem_type}\n' \
               f'stratify: {self.stratify}\n' \
               f'float_precision: {self.float_precision}'

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
        kfold = self.__make_folds()

        if self.problem_type == 'classification':
            return self.__classification(x, y, models, kfold)
        elif self.problem_type == 'regression':
            return self.__regression(x, y, models, kfold)
        else:
            print('Problem type must be passed (classification or regression)')

    def __make_folds(self):
        if self.stratify:
            return StratifiedKFold(n_splits=self.k_folds)
        else:
            return KFold(n_splits=self.k_folds)

    def __make_report(self, *metrics_list: any) -> list:
        scores = []
        for metric in metrics_list:
            scores.append(np.round(np.mean(metric), self.float_precision))
            scores.append(np.round(np.std(metric), self.float_precision))
        return scores

    def __regression(self, x: any, y: any, models: list, kfold: any) -> DataFrame:
        scores_df = DataFrame(columns=[
            'mean_r2', 'std_r2',
            'mean_mse', 'std_mse',
        ])

        for model in models:
            mse = []
            r2 = []

            for train, test in kfold.split(x, y):
                model.fit(x[train], y[train])
                predictions = model.predict(x[test])
                mse.append(mean_absolute_error(y[test], predictions))
                r2.append(r2_score(y[test], predictions))

            scores_df = scores_df.append(
                Series(
                    data=self.__make_report(mse, r2),
                    index=scores_df.columns,
                    name=model.__class__.__name__
                )
            )

        return scores_df

    def __classification(self, x: any, y: any, models: list, kfold: any) -> DataFrame:
        scores_df = DataFrame(columns=[
            'mean_accuracy', 'std_accuracy',
            'mean_recall', 'std_recall',
            'mean_precision', 'std_precision'
        ])

        for model in models:
            accuracy = []
            recall = []
            precision = []

            for train, test in kfold.split(x, y):
                model.fit(x[train], y[train])
                predictions = model.predict(x[test])
                accuracy.append(accuracy_score(y[test], predictions))
                recall.append(recall_score(y[test], predictions))
                precision.append(precision_score(y[test], predictions, zero_division=1))

            scores_df = scores_df.append(
                Series(
                    data=self.__make_report(accuracy, recall, precision),
                    index=scores_df.columns,
                    name=model.__class__.__name__
                )
            )

        return scores_df
