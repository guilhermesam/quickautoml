import numpy as np


class LogisticRegressionStub:
    def __init__(self, learning_rate=0.01, minimum_error=0.1, fit_intercept=True):
        self.learning_rate = learning_rate
        self.minimum_error = minimum_error
        self.loss = []
        self.theta = None
        self.fit_intercept = fit_intercept

    def __str__(self):
        return f'LogisticRegressionStub(learning_rate={self.learning_rate}, fit_intercept={self.fit_intercept})'

    def get_params(self, deep=True):
        return {
            'learning_rate': self.learning_rate,
            'minimum_error': self.minimum_error,
            'fit_intercept': self.fit_intercept
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @staticmethod
    def add_intercept(X):
        theta0 = np.ones((X.shape[0]))
        return np.insert(X, 0, theta0, axis=1)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, theta):
        z = np.dot(X, theta)
        step1 = y * np.log(self.sigmoid(z))
        step2 = (1 - y) * np.log(1 - self.sigmoid(z))
        return -sum(step1 + step2) / len(X)

    def fit(self, X, y, iterations=30, annotation=False):
        if self.fit_intercept:
            X = self.add_intercept(X)

        self.theta = np.zeros(X.shape[1])
        m = len(X)

        for iteration in range(iterations):
            z = np.dot(X, self.theta)
            hypothesis = self.sigmoid(z)
            gradient = np.dot(X.T, hypothesis - y) / m
            self.theta -= self.learning_rate * gradient
            loss = self.cost_function(X, y, self.theta)
            self.loss.append(loss)

            if loss <= self.minimum_error:
                break

            if annotation and iteration % 100 == 0:
                print('Loss in step {} = {}'.format(iteration, loss))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)

        return self.sigmoid(np.dot(X, self.theta))


class ModelStub:
    def __init__(self, parameter1=1, parameter2=1):
        self.parameter1 = parameter1
        self.parameter2 = parameter2

    def get_params(self, deep=True):
        return {
            'parameter1': self.parameter1,
            'parameter2': self.parameter2,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        pass

    def predict(self, X):
        return [1] * len(X[0])
