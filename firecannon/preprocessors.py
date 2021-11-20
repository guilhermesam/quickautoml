from abc import ABC, abstractmethod
from typing import List, Union

from numpy import ndarray, array
from pandas import DataFrame


class AbstractStep(ABC):
  @abstractmethod
  def run(self, matrix: Union[ndarray, List[list], DataFrame]):
    pass


class ConcreteStep(AbstractStep):
  def run(self, matrix: Union[ndarray, List[list], DataFrame]):
    return matrix


class StepDecorator(AbstractStep):
  def __init__(self, step: AbstractStep) -> None:
    self._step = step

  @property
  def step(self) -> AbstractStep:
    return self._step

  def run(self, matrix: Union[ndarray, List[list], DataFrame]):
    return self._step.run(matrix)


class GenericMatrixToDataframeDecorator(StepDecorator):
  def run(self, matrix: Union[ndarray, List[list], DataFrame]):
    if not isinstance(matrix, DataFrame):
      matrix = DataFrame(matrix)
    return self.step.run(matrix)


class RemoveNullValuesDecorator(StepDecorator):
  def run(self, matrix: Union[ndarray, List[list], DataFrame]):
    matrix = matrix.dropna(axis=1, how='all')
    matrix = matrix.dropna()
    return self.step.run(matrix)


class RemoveDuplicatesDecorator(StepDecorator):
  def run(self, matrix: Union[ndarray, List[list], DataFrame]):
    matrix = matrix.drop_duplicates()
    return self.step.run(matrix)


class GenericMatrixToNPArrayDecorator(StepDecorator):
  def run(self, matrix: Union[ndarray, List[list], DataFrame]):
    if not isinstance(matrix, ndarray):
      matrix = array(matrix)
    return self.step.run(matrix)


class DataPreprocessor:
  def __init__(self):
    pass

  @staticmethod
  def __initialize_steps(step: AbstractStep, matrix: any) -> None:
    return step.run(matrix)

  def run(self, matrix: Union[ndarray, List[list], DataFrame]) -> object:
    first_step = ConcreteStep()
    convert_matrix_to_array_step = GenericMatrixToNPArrayDecorator(first_step)
    remove_duplicates_step = RemoveDuplicatesDecorator(convert_matrix_to_array_step)
    remove_null_step = RemoveNullValuesDecorator(remove_duplicates_step)
    convert_matrix_to_dataframe_step = GenericMatrixToDataframeDecorator(remove_null_step)
    return self.__initialize_steps(convert_matrix_to_dataframe_step, matrix)
