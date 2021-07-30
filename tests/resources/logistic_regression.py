import numpy as np


def add_intercept(X):
    """
    Adiciona os valores de theta0 na primeira posição da matriz
    :param X: um array contendo as variáveis explicativas
    :return: o arrat X, com a nova coluna de 1's adicionada na posição 0
    """
    theta0 = np.ones((X.shape[0]))
    return np.insert(X, 0, theta0, axis=1)


def sigmoid(z):
    """
    Função logística da regressão:
    :param: a função linear W transposto X
    :return: a hipótese da função sigmoid
    """
    return 1 / (1 + np.exp(-z))


def cost_function(X, y, theta):
    """
    Função de custo para determinar o erro em
    cada etapa de treinamento
    :param X: o array contendo as variáveis explicativas
    :param y: o array contendo a variável target
    :param theta: o array de pesos theta
    :return: o valor do custo dados os parâmetros
    """
    z = np.dot(X, theta)
    step1 = y * np.log(sigmoid(z))
    step2 = (1 - y) * np.log(1 - sigmoid(z))
    return -sum(step1 + step2) / len(X)


class LogisticRegression(object):
    """
    Classe que implementa um modelo de regressão logística
    Parâmetros
    ----------
    learning_rate : float, default=0.01
        Usado para especificar o tamanho da passada em cada etapa de
        treinamento
    minimum_error : float, default=0.1
        Define o valor mínimo de erro tolerado pelo modelo, o que irá
        interromper o treinamento caso seja alcançado antes de todas as
        etapas serem executadas
    fit_intercept : bool, default=True
        Especifica se uma constante (por exemplo, bias ou intercept) será
        adicionada na função de decisão
    """

    def __init__(self, learning_rate=0.01, minimum_error=0.1, fit_intercept=True):
        self.learning_rate = learning_rate
        self.minimum_error = minimum_error
        self.loss = []
        self.theta = None
        self.fit_intercept = fit_intercept

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

    def fit(self, X, y, iterations=30, annotation=False):
        """
        :param X: o array contendo as variáveis explicativas
        :param y: o array contendo a variável target
        :param iterations: o número máximo de iterações que o modelo irá realizar
        :param annotation: caso seja definido como true, printa na tela o custo a
        cada 100 etapas de treinamento
        O objetivo desta função é atualizar os valores de theta de modo a reduzir ao
        máximo o erro do modelo
        """
        if self.fit_intercept:
            X = add_intercept(X)

        self.theta = np.zeros(X.shape[1])
        m = len(X)

        for iteration in range(iterations):
            z = np.dot(X, self.theta)
            hypothesis = sigmoid(z)
            gradient = np.dot(X.T, hypothesis - y) / m
            self.theta -= self.learning_rate * gradient
            loss = cost_function(X, y, self.theta)
            self.loss.append(loss)

            if loss <= self.minimum_error:
                break

            if annotation and iteration % 100 == 0:
                print('Loss in step {} = {}'.format(iteration, loss))

    def predict(self, X, threshold=0.5):
        """
        Baseado na probabilidade da regressão, a função retorna em qual classe
        aquele ponto se encontra
        :param X: o array contendo as variáveis explicativas
        :param threshold: o valor do limite de decisão
        :return: o vetor de classes para o dataset X
        """
        return self.predict_proba(X) >= threshold

    def predict_proba(self, X):
        """
        Define a probabilidade de um ponto pertencer a determinada classe
        :param X: o array contendo as variáveis explicativas
        :return: o valor da probabilidade no intervalo {0,1}
        """
        if self.fit_intercept:
            X = add_intercept(X)

        return sigmoid(np.dot(X, self.theta))
