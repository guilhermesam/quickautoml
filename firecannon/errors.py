class ProblemTypeNotSuppliedException(Exception):
    """thrown when problem type (classification or regression) is not supplied"""
    def __str__(self):
        return f"pass 'classification' or 'regression' via test settings"


class IncompatibleDataShapeException(Exception):
    """thrown when shapes of X and y are incompatiple"""

    def __init__(self, X_rows: int, y_rows: int):
        self.message = f'X has {X_rows} rows, but y has {y_rows} rows'
        super(Exception, self).__init__(self.message)

    def __str__(self):
        return self.message
