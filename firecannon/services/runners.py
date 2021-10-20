from numpy import mean

from firecannon.services.best_hyperparams import BestParamsTestSuite
from firecannon.adapters import sklearn_models_adapters
from firecannon.presentation.reports import BarplotReport, CsvReport, DataframeReport


class BaseModelAgg:
    def __init__(self, metric: str, report_type: str = None, models_settings: str = None) -> None:
        if models_settings is None:
            models_settings = {}
        self.metric = metric
        self.k_folds = 5
        self.n_jobs = -1
        self.random_state = 777
        self.verbose = False
        self.scoring = None
        self.__best_model = None
        self.report_type = report_type
        self.__fitted = False

        if not models_settings:
            self.__default_models_config()
        else:
            self.model_settings = models_settings

    def __default_models_config(self):
        pass

    @staticmethod
    def make_report(report_type: str, scores: dict):
        report_types = {
            'dataframe': DataframeReport(),
            'plot': BarplotReport(),
            'csv': CsvReport()
        }
        report_types.get(report_type).make_report(scores)

    def get_best_model(self):
        return self.__best_model.__str__()

    @staticmethod
    def __extract_best_model(scores):
        models_by_metric = {}
        for model, param in scores.items():
            models_by_metric.update({model: mean(param)})

        return max(models_by_metric, key=models_by_metric.get)


class Regressor(BaseModelAgg):
    def __init__(self, metric: str, models_settings: dict = None, scoring: str = 'neg_mean_squared_error') -> None:
        super().__init__(metric, models_settings)
        self.scoring = scoring

    def __default_models_config(self):
        self.model_settings = {
            sklearn_models_adapters.RandomForestRegressor: {
                'n_estimators': [50, 100, 150],
                'criterion': ['mse', 'mae'],
                'max_features': ['auto', 'log2', 'sqrt'],
            },
            sklearn_models_adapters.Lasso: {
                'alpha': [1.0, 1.5, 2.0],
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            sklearn_models_adapters.ElasticNet: {
                'alpha': [1.0, 1.5, 2.0],
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        }

    def fit(self, X, y):
        best_hyperparams_test = BestParamsTestSuite(
            k_folds=self.k_folds,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            scoring=self.scoring
        )
        best_models = {}
        for model, params in self.model_settings.items():
            best_model = best_hyperparams_test.run(X, y, {model: params})
            best_models.update(best_model)

        return self.__extract_best_model(best_models)

    def predict(self, X_test):
        best_model = self.__best_model
        return best_model.predict(X_test)


class Classifier(BaseModelAgg):
    def __init__(self, metric: str = 'accuracy',
                 report_type: str = None,
                 models_settings: dict = None
                 ) -> None:
        super().__init__(metric, models_settings, report_type)
        if not models_settings:
            self.__default_models_config()
        else:
            self.model_settings = models_settings

    def __default_models_config(self):
        self.model_settings = {
            sklearn_models_adapters.KNeighborsClassifier(): {
                'n_neighbors': [3, 5, 7],
                'leaf_size': [15, 30, 45, 60],
                'weights': ['uniform', 'distance']
            },
            sklearn_models_adapters.RandomForestClassifier(): {
                'n_estimators': [50, 100, 150],
                'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'log2', 'sqrt'],
            },
            sklearn_models_adapters.AdaBoostClassifier(): {
                'n_estimators': [50, 100, 150],
                'learning_rate': [1, 0.1, 0.5]
            }
        }

    @staticmethod
    def __extract_best_model(scores):
        models_by_metric = {}
        for model, param in scores.items():
            models_by_metric.update({model: mean(param)})

        return max(models_by_metric, key=models_by_metric.get)

    def fit(self, X, y):
        best_hyperparams_test = BestParamsTestSuite(
            k_folds=self.k_folds,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            scoring=self.metric
        )

        scores = {}
        for model, params in self.model_settings.items():
            best_model = best_hyperparams_test.run(X, y, {model: params})
            scores.update(best_model)

        self.__best_model = self.__extract_best_model(scores)

        if self.report_type:
            self.make_report(self.report_type, scores)

    def predict(self, y):
        best_model = self.__best_model
        return best_model.predict(y)
