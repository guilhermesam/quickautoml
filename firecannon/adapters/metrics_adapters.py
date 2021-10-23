from firecannon.protocols import Metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error, r2_score


class SKLearnMetrics(Metrics):
    @staticmethod
    def mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def r2(y_true, y_pred):
        return r2_score(y_true, y_pred)

    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        return precision_score(y_true, y_pred)

    @staticmethod
    def recall(y_true, y_pred):
        return recall_score(y_true, y_pred)
