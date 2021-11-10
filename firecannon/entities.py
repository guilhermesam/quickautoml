from typing import Union


class NaiveModel:
  def __init__(self, name: str, estimator) -> None:
    self.name = name
    self.estimator = estimator

  def __str__(self):
    return self.estimator.__str__()


class FittedModel(NaiveModel):
  def __init__(self, name: str, cv_score: float, estimator) -> None:
    super().__init__(name, estimator)
    self.cv_score = cv_score

  def predict(self, X):
    return self.estimator.predict(X)


class Hyperparameter:
  def __init__(self,
               name: str,
               data_type: str,
               min_value: Union[int, float],
               max_value: Union[int, float]):
    self.name = name
    self.data_type = data_type
    self.min_value = min_value
    self.max_value = max_value
