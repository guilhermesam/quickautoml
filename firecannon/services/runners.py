from abc import ABC, abstractmethod

from firecannon.entities import Model
from firecannon.services.best_hyperparams import BestParamsTestSuite
from firecannon.adapters.models_adapters import SKLearnModelsSupplier, ModelsSupplier
from firecannon.reports import BarplotReport, CsvReport, JsonReport


class BaseModel:
    def __init__(self, metric: str, 
                    report_type: str = None,
                    models_settings: str = None,
                    metric_threshold: float = 0.80,
                    models_supplier: ModelsSupplier = SKLearnModelsSupplier()):
        self.metric = metric
        self.report_type = report_type
        self._models_supplier = models_supplier
        self.k_folds = 5
        self.n_jobs = -1
        self.random_state = 777
        self.verbose = False
        self.best_model = None
        self.metric_threshold = metric_threshold
        self.__valid_report_types = [
            'plot', 'csv', 'json'
        ]
        self.model_settings = self._default_models_config() if not models_settings else models_settings

    @abstractmethod
    def _default_models_config(self):
        pass

    def fit(self, X, y) -> None:
        best_hyperparams_finder = BestParamsTestSuite(
            k_folds=self.k_folds,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            scoring=self.metric
        )

        scores = {}

        for model, params in self.model_settings.items():
            best_model: Model = best_hyperparams_finder.run(X, y, model, params)
            scores.update({best_model.name: best_model.score})
            if best_model.score > self.metric_threshold:
                self.best_model = best_model
                break   
        else:
            self.best_model = self._extract_best_model(scores)

        if self.__conditions_to_make_report():
            self.make_report(self.report_type, scores)

    def predict(self, X):
        return self.best_model.predict(X)

    def __conditions_to_make_report(self) -> bool:
        return self.report_type and (self.report_type in self.__valid_report_types)

    def make_report(self, report_type: str, scores: dict):
        report_types = {
            'plot': BarplotReport(),
            'csv': CsvReport(),
            'json': JsonReport()
        }
        report_types.get(report_type).make_report(scores)

    @staticmethod
    def _extract_best_model(scores):
        return max(scores, key=scores.get)


class Regressor(BaseModel):
    def __init__(self, 
                metric: str = 'r2',
                report_type: str = None,
                models_settings: dict = None
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


class Classifier(BaseModel):
    def __init__(self, 
                metric: str = 'accuracy',
                report_type: str = None,
                models_settings: dict = None
                ) -> None:
        super().__init__(metric, report_type, models_settings)
        self.report_type = report_type

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
