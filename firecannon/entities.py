class Model:
  def __init__(self, name: str, score: float, estimator: any) -> None:
    self.name = name
    self.score = score
    self.estimator = estimator

  def predict(self, X):
    return self.estimator.predict(X)
