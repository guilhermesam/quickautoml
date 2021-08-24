from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error, r2_score


class RegressionMetrics:
    @staticmethod
    def mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def r2_score(y_true, y_pred):
        return r2_score(y_true, y_pred)


class ClassificationMetrics:
    @staticmethod
    def accuracy_score(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision_score(y_true, y_pred):
        return precision_score(y_true, y_pred)

    @staticmethod
    def recall_score(y_true, y_pred):
        return recall_score(y_true, y_pred)
