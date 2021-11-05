class Model:
  def __init__(self, name: str, cv_score: float, estimator: any) -> None:
    self.name = name
    self.cv_score = cv_score
    self.estimator = estimator

  def predict(self, X):
    return self.estimator.predict(X)
