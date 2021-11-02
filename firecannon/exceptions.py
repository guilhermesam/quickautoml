class IncompatibleDataShapeException(Exception):
    def __init__(self, X_rows: int, y_rows: int):
        self.message = f'X has {X_rows} rows, but y has {y_rows} rows'
        super(Exception, self).__init__(self.message)

    def __str__(self):
        return self.message

class InvalidParamException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message
