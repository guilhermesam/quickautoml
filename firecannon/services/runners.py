from numpy import mean

from firecannon.services.best_hyperparams import BestParamsTestSuite

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso, ElasticNet


class BaseModelAgg:
    def __init__(self) -> None:
        self.k_folds = 5
        self.n_jobs = -1
        self.random_state = 777
        self.verbose = False
        self.scoring = None
        self.best_model = None
        self.__fitted = False

    @property
    def get_best_model(self):
        return self.best_model.__class__.__name__

    @staticmethod
    def __extract_best_model(scores):
        models_by_metric = {}
        for model, param in scores.items():
            models_by_metric.update({model: mean(param)})

        return max(models_by_metric, key=models_by_metric.get)


class Regressor(BaseModelAgg):
    def __init__(self) -> None:
        super().__init__()
        self.scoring = 'neg_mean_squared_error'

    def fit(self, X, y):
        rf = RandomForestRegressor()
        lasso = Lasso()
        en = ElasticNet()

        model_settings = {
            rf: {
                'n_estimators': [50, 100, 150],
                'criterion': ['mse', 'mae'],
                'max_features': ['auto', 'log2', 'sqrt'],
            },
            lasso: {
                'alpha': [1.0, 1.5, 2.0],
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            en: {
                'alpha': [1.0, 1.5, 2.0],
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        }

        best_hyperparams_test = BestParamsTestSuite(
            k_folds=self.k_folds,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            scoring=self.scoring
        )
        best_models = {}
        for model, params in model_settings.items():
            best_model = best_hyperparams_test.run(X, y, {model: params})
            best_models.update(best_model)

        self.best_model = self.__extract_best_model(best_models)
        return self.best_model

    def predict(self, X_test):
        best_model = self.best_model
        return best_model.predict(y)


class Classifier(BaseModelAgg):
    def __init__(self) -> None:
        super().__init__()
        self.scoring = 'accuracy'

    @staticmethod
    def __extract_best_model(scores):
        models_by_metric = {}
        for model, param in scores.items():
            models_by_metric.update({model: mean(param)})

        return max(models_by_metric, key=models_by_metric.get)

    def fit(self, X, y):
        rf = RandomForestClassifier(random_state=self.random_state)
        knn = KNeighborsClassifier()
        ada = AdaBoostClassifier(random_state=self.random_state)

        model_settings = {
            knn: {
                'n_neighbors': [3, 5, 7],
                'leaf_size': [15, 30, 45, 60],
                'weights': ['uniform', 'distance']
            },
            rf: {
                'n_estimators': [50, 100, 150],
                'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'log2', 'sqrt'],
            },
            ada: {
                'n_estimators': [50, 100, 150],
                'learning_rate': [1, 0.1, 0.5]
            }
        }

        best_hyperparams_test = BestParamsTestSuite(
            k_folds=self.k_folds,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            scoring=self.scoring
        )

        best_models = {}
        for model, params in model_settings.items():
            best_model = best_hyperparams_test.run(X, y, {model: params})
            best_models.update(best_model)

        self.best_model = self.__extract_best_model(best_models)
        return self.best_model

    def predict(self, y):
        best_model = self.best_model
        return best_model.predict(y)
