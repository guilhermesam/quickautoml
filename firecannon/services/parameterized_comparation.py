from sklearn.model_selection import StratifiedKFold, KFold
from numpy import mean

from firecannon.utils import check_shape_compatibility, convert_to_np_array
from firecannon.errors import IncompatibleDataShapeException, ProblemTypeNotSuppliedException
from firecannon.presentation.reports import BarplotReport, DataframeReport, CsvReport
from firecannon.infra.metrics_adapters import RegressionMetrics, ClassificationMetrics


class ParameterizedTestSuite:
    """
    Class to test entities with tuned hyperparameters
    """

    def __init__(self, problem_type,
                 k_folds,
                 stratify,
                 float_precision,
                 metric,
                 ):
        self.problem_type = problem_type
        self.k_folds = k_folds
        self.stratify = stratify
        self.float_precision = float_precision
        self.metric = metric
        self.best_model = None
        self.__scores = {}

    def __str__(self):
        return f'k_folds: {self.k_folds}\n' \
               f'problem_type: {self.problem_type}\n' \
               f'stratify: {self.stratify}\n' \
               f'float_precision: {self.float_precision}' \
               f'metric: {self.metric}'

    def run(self, x: any, y: any, models: list):
        """
        Parameters
        ----------
        x: any implementation of 2D matrix with features for training
        y: any implementation of 2D matrix with labels
        models: list of instantiated entities
        -------
        """
        if not check_shape_compatibility(x, y):
            raise IncompatibleDataShapeException(x.shape[0], y.shape[0])

        x = convert_to_np_array(x)
        y = convert_to_np_array(y)
        kfold = self.__make_folds()

        report_type = {
            'dataframe': DataframeReport(),
            'plot': BarplotReport(),
            'csv': CsvReport()
        }

        if self.problem_type == 'classification':
            scores = self.__classification(x, y, models, kfold)
            # report_type.get(self.report_type).make_report(scores)
            return scores

        elif self.problem_type == 'regression':
            scores = self.__regression(x, y, models, kfold)
            return scores
            # return report_type.get(self.report_type).make_report(scores)
        else:
            raise ProblemTypeNotSuppliedException('Problem type must be passed (classification or regression)')

    def __make_folds(self):
        if self.stratify:
            return StratifiedKFold(n_splits=self.k_folds)
        else:
            return KFold(n_splits=self.k_folds)

    def __regression(self, x: any, y: any, models: list, kfold: any) -> dict:
        scores = {}
        metrics = {
            'mse': RegressionMetrics.mse,
            'r2_score': RegressionMetrics.r2_score
        }
        current_metric = metrics.get(self.metric)

        for model in models:
            metrics_for_fold = []

            for train, test in kfold.split(x, y):
                model.fit(x[train], y[train])
                predictions = model.predict(x[test])
                metrics_for_fold.append(current_metric(y[test], predictions))

            scores.update({
                model: mean(metrics_for_fold)
            })

        return scores

    def __classification(self, x: any, y: any, models: list, kfold: any) -> dict:
        scores = {}
        metrics = {
            'accuracy': ClassificationMetrics.accuracy_score,
            'recall': ClassificationMetrics.recall_score,
            'precision': ClassificationMetrics.precision_score
        }
        current_metric = metrics.get(self.metric)

        for model in models:
            metrics_for_fold = []

            for train, test in kfold.split(x, y):
                model.fit(x[train], y[train])
                predictions = model.predict(x[test])
                metrics_for_fold.append(current_metric(y[test], predictions))

            scores.update({
                model: mean(metrics_for_fold)
            })

        return scores


