from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso, ElasticNet


class ModelsSupplier(ABC):
  def __init__(self) -> None:
    self._models_codes = None

  @abstractmethod
  def get_model(model_name: str):
    """
    Deve retornar uma instância de um modelo, de acordo com o nome do mesmo. Possíveis nomes:
    Classificadores:
    knn-c: KNearestNeighbors Classifier
    rf-c: RandomForest Classifier
    ada-c: AdaBoost Classifier

    knn-r: KNearestNeighbors Regressor
    rf-r: RandomForest Regressor
    lasso: Lasso Regressor
    en: ElasticNet Regressor
    """
    pass


class SKLearnModelsSupplier(ModelsSupplier):
  def __init__(self) -> None:
    super().__init__()
    self._models_codes = {
      'knn-c': KNeighborsClassifier(),
      'rf-c': RandomForestClassifier(),
      'ada-c': AdaBoostClassifier(),
      'rf-r': RandomForestRegressor(),
      'lasso': Lasso(),
      'en': ElasticNet()
    }

  def get_model(self, model_name: str):
    required_model = self._models_codes.get(model_name)
    return required_model
