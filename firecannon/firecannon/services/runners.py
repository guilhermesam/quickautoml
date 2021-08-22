from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error

from firecannon.services.best_hyperparams import BestParamsTestSuite
# from firecannon.presentation.colors import ConsoleColors
from firecannon.errors import ModelNotFittedException

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from numpy import mean


class BaseModel:
    def __init__(self):
        self.__fitted = False
        self.__best_model = None

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_best_model(self):
        if not self.__fitted:
            raise ModelNotFittedException()
        else:
            return self.__best_model


class Regressor(BaseModel):
    def __init__(self,
                 k_folds=4,
                 n_jobs=-1,
                 verbose=False,
                 output_path='NO_PATH',
                 scoring='neg_mean_squared_error'):
        super().__init__()
        self.k_folds = k_folds
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.output_path = output_path
        self.scoring = scoring

    def get_params(self, deep=True):
        return {
            'k_folds': self.k_folds,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
            'output_path': self.output_path,
            'scoring': self.scoring
        }

    def fit(self, X, y):
        lasso = Lasso()
        ridge = Ridge()
        adaboost = AdaBoostRegressor()

        model_settings = {
            lasso: {
                'fit_intercept': [True, False],
                'selection': ['cyclic', 'random'],
                'alpha': [1.0, 0.5, 0.1]
            },
            ridge: {
                'fit_intercept': [True, False],
                'solver': ['auto', 'svd', 'sag', 'lsqr'],
                'alpha': [1.0, 0.5, 0.1]
            },
            adaboost: {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [1, 0.5, 0.1],
                'loss': ['linear', 'square', 'exponential']
            }
        }

        # print('Buscando melhores hiperparâmetros...', end='')
        best_hyperparams_test = BestParamsTestSuite(
            self.k_folds,
            self.n_jobs,
            self.verbose,
            self.output_path,
            self.scoring
        )
        best_models = {}
        for model, params in model_settings.items():
            best_model = best_hyperparams_test.run(X, y, {model: params})
            best_models.update(best_model)

        print(best_models)

        self.__best_model = self.extract_best_model(best_models)
        self.__best_model.fit(X, y)
        print(self.__best_model)
        self.__fitted = True
        # print(f'{ConsoleColors.OKGREEN}OK{ConsoleColors.END_LINE}')

    def predict(self, y):
        if not self.__fitted:
            raise ModelNotFittedException()
        else:
            return self.__best_model.predict(y)

    @staticmethod
    def extract_best_model(models):
        """
        :param dict models: dict of format {
            model_1: [value_1, value_2],
            model_2: [value_3, value_4]
        }
        each value is a result of metric for each fold in kfolds
        """
        best_performance = min(models.values())
        best_model = None
        for model, metrics in models.items():
            mean_metric = mean(metrics)
            if mean_metric > best_performance:
                best_performance = mean_metric
                best_model = model
        return best_model


class Classifier(BaseModel):
    def __init__(self,
                 k_folds=4,
                 n_jobs=-1,
                 verbose=False,
                 output_path='NO_PATH',
                 scoring='accuracy'):
        super().__init__()
        self.k_folds = k_folds
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.output_path = output_path
        self.scoring = scoring

    def get_params(self, deep=True):
        return {
            'k_folds': self.k_folds,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
            'output_path': self.output_path,
            'scoring': self.scoring
        }

    def fit(self, X, y):
        rf = RandomForestClassifier()
        knn = KNeighborsClassifier()
        ada = AdaBoostClassifier()

        model_settings = {
            knn: {
                'n_neighbors': [3, 5, 7],
                'leaf_size': [15, 30, 45],
                'weights': ['uniform', 'distance']
            },
            rf: {
                'n_estimators': [100, 200, 300],
                'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'sqrt'],
            },
            ada: {
                'n_estimators': [50, 100, 150],
                'learning_rate': [1, 0.1, 0.5]
            }
        }

        # print('Buscando melhores hiperparâmetros...', end='')
        best_hyperparams_test = BestParamsTestSuite(
            self.k_folds,
            self.n_jobs,
            self.verbose,
            self.output_path,
            self.scoring
        )
        best_models = {}
        for model, params in model_settings.items():
            best_model = best_hyperparams_test.run(X, y, {model: params})
            best_models.update(best_model)

        self.__best_model = self.extract_best_model(best_models)
        self.__best_model.fit(X, y)
        self.__fitted = True
        # print(f'{ConsoleColors.OKGREEN}OK{ConsoleColors.END_LINE}')

    def predict(self, y):
        if not self.__fitted:
            raise ModelNotFittedException()
        else:
            return self.__best_model.predict(y)

    @staticmethod
    def extract_best_model(models):
        """
        :param dict models: dict of format {
            model_1: [value_1, value_2],
            model_2: [value_3, value_4]
        }
        each value is a result of metric for each fold in kfolds
        """
        best_performance = 0
        best_model = None
        for model, metrics in models.items():
            mean_metric = mean(metrics)
            if mean_metric > best_performance:
                best_performance = mean_metric
                best_model = model
        return best_model


if __name__ == '__main__':
    from pandas import read_csv
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston
    from sklearn.metrics import precision_score, mean_squared_error
    from sklearn.svm import SVC

    X, y = load_boston(return_X_y=True)
    # df = read_csv('selected_features.csv')
    # X = df.drop('class', axis=1)
    # y = df['class']
    regressor = Regressor()
    baseline = SGDRegressor()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    regressor.fit(X_train, y_train)
    # svc_model.fit(X_train, y_train)

    # svc_predictions = svc_model.predict(X_test)
    pred = regressor.predict(X_test)

    print(f'Classifier Accuracy: {round(mean_squared_error(y_test, pred), 4)}')


"""
print('Buscando o melhor modelo...', end='')
        best_models_test = ParameterizedTestSuite(
            {
                'k_folds': 3,
                'float_precision': 3,
                'problem_type': 'classification',
                'stratify': True,
                'report_type': 'csv',
                'metric': 'accuracy'
            }
        )
        scores = best_models_test.run(X, y, best_models)
        self.best_model = best_models_test.get_best_model(scores)
"""