class Metrics:
    def mse(self, y_true, y_pred):
        raise NotImplementedError('mse')

    def r2(self, y_true, y_pred):
        raise NotImplementedError('r2')

    def accuracy(self, y_true, y_pred):
        raise NotImplementedError('accuracy')

    def precision(self, y_true, y_pred):
        raise NotImplementedError('precision')

    def recall(self, y_true, y_pred):
        raise NotImplementedError('recall')
