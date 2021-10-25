from numpy import mean
from abc import ABC, abstractmethod

from firecannon.services.best_hyperparams import BestParamsTestSuite
from firecannon.adapters.models_adapters import SKLearnModelsSupplier, ModelsSupplier
from firecannon.reports import BarplotReport, CsvReport, DataframeReport


class BaseModelAgg:
    def __init__(self, metric: str, 
                    report_type: str = None,
                    models_settings: str = None,
                    models_supplier: ModelsSupplier = SKLearnModelsSupplier()):
        self.metric = metric
        self._models_supplier = models_supplier
        self.k_folds = 5
        self.n_jobs = -1
        self.random_state = 777
        self.verbose = False
        self.best_model = None
        self.report_type = report_type
        self.__valid_report_types = [
            'plot', 'csv', 'json'
        ]

    @abstractmethod
    def _default_models_config(self):
        pass

    def make_report(self, report_type: str, scores: dict):
        if report_type not in self.__valid_report_types:
            raise ValueError('Defina um método de relatório válido (\'plot\' para gerar ' +
            'um gráfico ou \'csv\' para gerar uma planilha)')
        report_types = {
            'plot': BarplotReport(),
            'csv': CsvReport()
        }
        report_types.get(report_type).make_report(scores)

    def get_best_model(self):
        return self.best_model


    @staticmethod
    def _extract_best_model(scores):
        models_by_metric = {}
        for model, param in scores.items():
            models_by_metric.update({model: mean(param)})

        return max(models_by_metric, key=models_by_metric.get)


class Regressor(BaseModelAgg):
    def __init__(self, report_type: str,
                models_settings: dict,
                models_supplier: list,
                metric: str = 'r2'
                ) -> None:
        super().__init__(metric, models_settings, report_type)

    def __default_models_config(self):
        self.model_settings = {
            self._models_supplier.RandomForestRegressor: {
                'n_estimators': [50, 100, 150],
                'criterion': ['mse', 'mae'],
                'max_features': ['auto', 'log2', 'sqrt'],
            },
            self._models_supplier.Lasso: {
                'alpha': [1.0, 1.5, 2.0],
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            self._models_supplier.ElasticNet: {
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

        self.best_model = self._extract_best_model(best_models)

    def predict(self, X_test):
        return self.best_model.predict(X_test)


class Classifier(BaseModelAgg):
    def __init__(self, metric: str = 'accuracy',
                 report_type: str = None,
                 models_settings: dict = None
                 ) -> None:
        super().__init__(metric, models_settings, report_type)
        self.report_type = report_type
        self.model_settings = self._default_models_config() if not models_settings else models_settings

    def _default_models_config(self):
        return {
            self._models_supplier.get_model('knn-c'): {
                'n_neighbors': [3, 5, 7],
                'leaf_size': [15, 30, 45, 60],
                'weights': ['uniform', 'distance']
            },
            self._models_supplier.get_model('rf-c'): {
                'n_estimators': [50, 100, 150],
                'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'log2', 'sqrt'],
            },
            self._models_supplier.get_model('ada-c'): {
                'n_estimators': [50, 100, 150],
                'learning_rate': [1, 0.1, 0.5]
            }
        }

    def fit(self, X, y):
        best_hyperparams_test = BestParamsTestSuite(
            k_folds=self.k_folds,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            scoring=self.metric
        )

        scores = {}

        for model, params in self.model_settings.items():
            best_model = best_hyperparams_test.run(X, y, model, params)
            scores.update(best_model)

        self.best_model = self._extract_best_model(scores)

        if self.report_type is not None:
            self.make_report(self.report_type, scores)

    def predict(self, y):
        return self.best_model.predict(y)


from sklearn.datasets import make_classification

X_c, y_c = make_classification(n_classes=2, n_features=5, n_samples=100)

c = Classifier(report_type='pltot')
c.fit(X_c, y_c)
