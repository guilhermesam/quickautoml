from numpy import ndarray, array
from pandas import DataFrame


class Component:
  def run(self, matrix):
    pass


class ConcreteComponent(Component):
  def run(self, matrix):
    return matrix


class Decorator(Component):
  def __init__(self, component: Component) -> None:
    self._component = component

  @property
  def component(self) -> Component:
    return self._component

  def run(self, matrix):
    return self._component.run(matrix)


class GenericMatrixToDataframeDecorator(Decorator):
  def run(self, matrix):
    matrix = DataFrame(matrix)
    return self.component.run(matrix)


class RemoveNullValuesDecorator(Decorator):
  def run(self, matrix):
    for col in matrix.columns:
      if matrix[col].isnull().all():
        matrix.drop(col)
    matrix = matrix.dropna()
    return self.component.run(matrix)


class GenericMatrixToNPArrayDecorator(Decorator):
  def run(self, matrix):
    if not isinstance(matrix, ndarray):
      matrix = array(matrix)
    return self.component.run(matrix)


class DataPreprocessor:
  def __init__(self):
    pass

  @staticmethod
  def __client_code(component: Component, matrix: any) -> None:
    return component.run(matrix)

  def run(self, matrix: any) -> object:
    simple = ConcreteComponent()
    decorator1 = GenericMatrixToNPArrayDecorator(simple)
    decorator2 = RemoveNullValuesDecorator(decorator1)
    decorator3 = GenericMatrixToDataframeDecorator(decorator2)
    return self.__client_code(decorator3, matrix)
