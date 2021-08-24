from sklearn.model_selection import StratifiedKFold, KFold
from numpy import mean

from firecannon.utils import check_shape_compatibility, convert_to_np_array
from firecannon.errors import IncompatibleDataShapeException, ProblemTypeNotSuppliedException
from firecannon.presentation.reports import BarplotReport, DataframeReport, CsvReport
from firecannon.protocols.metrics import RegressionMetrics, ClassificationMetrics


class ParameterizedTestSuite:
    """
    Class to test entities with tuned hyperparameters
    """

    def __init__(self, test_settings: dict = None):
        if test_settings is None:
            self.test_settings = {}
        self.problem_type = test_settings.get('problem_type')
        self.k_folds = test_settings.get('k_folds') or 4
        self.stratify = test_settings.get('stratify') or False
        self.float_precision = test_settings.get('float_precision') or 3
        self.report_type = test_settings.get('report_type') or 'dataframe'
        self.metric = test_settings.get('metric') or 'accuracy'

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
            return report_type.get(self.report_type).make_report(scores)
        else:
            raise ProblemTypeNotSuppliedException('Problem type must be passed (classification or regression)')

    def __make_folds(self):
        if self.stratify:
            return StratifiedKFold(n_splits=self.k_folds)
        else:
            return KFold(n_splits=self.k_folds)

    @staticmethod
    def __regression(x: any, y: any, models: list, kfold: any) -> dict:
        scores = {}

        for model in models:
            mse = []
            r2 = []

            for train, test in kfold.split(x, y):
                model.fit(x[train], y[train])
                predictions = model.predict(x[test])
                mse.append(RegressionMetrics.mse(y[test], predictions))
                r2.append(RegressionMetrics.r2_score(y[test], predictions))

            scores.update({
                model: {
                    'r2': r2,
                    'mse': mse
                }
            })

        return scores

    def __classification(self, x: any, y: any, models: list, kfold: any) -> dict:
        scores = {}
        metrics = {
            'accuracy': ClassificationMetrics.accuracy_score,
            'recall': ClassificationMetrics.recall_score,
            'precision': ClassificationMetrics.precision_score
        }

        for model in models:
            metric_values = []
            current_metric = metrics.get(self.metric)

            for train, test in kfold.split(x, y):
                model.fit(x[train], y[train])
                predictions = model.predict(x[test])
                metric_values.append(current_metric(y[test], predictions))

            scores.update({
                model: metric_values
            })

        print(scores)
        return scores

    def get_best_model(self, scores):
        models_by_metric = {}
        for model, param in scores.items():
            models_by_metric.update({model: mean(param)})

        return max(models_by_metric, key=models_by_metric.get)
