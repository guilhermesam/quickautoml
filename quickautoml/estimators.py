from quickautoml.hyperparameter_optimizer import HyperparamsOptimizer
from quickautoml.exceptions import InvalidParamException, ModelNotFittedException
from quickautoml.feature_engineering import FeatureEngineer
from quickautoml.preprocessors import DataPreprocessor
from quickautoml.entities import FittedModel


class Classifier:
    def __init__(self, metric: str):
        self.metric = metric
        self.data_preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.hyperparameter_optimizer = HyperparamsOptimizer(metric=self.metric)
        self.best_model = None

        self.__valid_metrics = ['accuracy', 'recall', 'f1_score']

        self.__classifier_run_validations()

    def _check_valid_metric(self) -> None:
        if self.metric not in self.__valid_metrics:
            raise InvalidParamException(f'Supplied metric is invalid. Choose a value from {self.__valid_metrics}')

    @property
    def cv_score(self):
        if not self.best_model:
            raise ModelNotFittedException("Estimator not fitted yet. Call fit method before")

    def __classifier_run_validations(self):
        self._check_valid_metric()

    def fit(self, train_data, labels) -> None:
        best_model: FittedModel = self.hyperparameter_optimizer.run(train_data, labels)
        self.best_model = best_model
        self.best_model.estimator.fit(train_data, labels)

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def prepare_data(self, data):
        processed_data = self.data_preprocessor.run(data)

        feat_engineered_data = self.feature_engineer.run(processed_data)
        del processed_data

        return feat_engineered_data


def make_classifier(metric: str):
    return Classifier(
        metric=metric
    )
