from abc import ABC, abstractmethod

class Metrics(ABC):
    @abstractmethod
    def mse(y_true, y_pred):
        pass
    
    @abstractmethod
    def r2(y_true, y_pred):
        pass

    @abstractmethod
    def accuracy(y_true, y_pred):
        pass

    @abstractmethod
    def precision(y_true, y_pred):
        pass

    @abstractmethod
    def recall(self, y_true, y_pred):
        pass
