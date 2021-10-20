from firecannon.protocols.metrics import Metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error, r2_score


class SKLearnMetrics(Metrics):
    def mse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def r2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def precision(self, y_true, y_pred):
        return precision_score(y_true, y_pred)

    def recall(self, y_true, y_pred):
        return recall_score(y_true, y_pred)
