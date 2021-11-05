from abc import ABC, abstractmethod

from numpy import ndarray, array
from pandas import DataFrame


class AbstractStep(ABC):
  @abstractmethod
  def run(self, matrix):
    pass


class ConcreteStep(AbstractStep):
  def run(self, matrix):
    return matrix


class StepDecorator(AbstractStep):
  def __init__(self, step: AbstractStep) -> None:
    self._step = step

  @property
  def step(self) -> AbstractStep:
    return self._step

  def run(self, matrix):
    return self._step.run(matrix)


class GenericMatrixToDataframeDecorator(StepDecorator):
  def run(self, matrix):
    if not isinstance(matrix, DataFrame):
      matrix = DataFrame(matrix)
    return self.step.run(matrix)


class RemoveNullValuesDecorator(StepDecorator):
  def run(self, matrix):
    for col in matrix.columns:
      if matrix[col].isnull().all():
        matrix.drop(col)
    matrix.dropna(inplace=True)
    return self.step.run(matrix)


class RemoveDuplicatesDecorator(StepDecorator):
  def run(self, matrix):
    matrix.drop_duplicates(inplace=True)
    return self.step.run(matrix)


class GenericMatrixToNPArrayDecorator(StepDecorator):
  def run(self, matrix):
    if not isinstance(matrix, ndarray):
      matrix = array(matrix)
    return self.step.run(matrix)


class DataPreprocessor:
  def __init__(self):
    pass

  @staticmethod
  def __initialize_steps(step: AbstractStep, matrix: any) -> None:
    return step.run(matrix)

  def run(self, matrix: any) -> object:
    first_step = ConcreteStep()
    convert_matrix_to_array_step = GenericMatrixToNPArrayDecorator(first_step)
    remove_duplicates_step = RemoveDuplicatesDecorator(convert_matrix_to_array_step)
    remove_null_step = RemoveNullValuesDecorator(remove_duplicates_step)
    convert_matrix_to_dataframe_step = GenericMatrixToDataframeDecorator(remove_null_step)
    return self.__initialize_steps(convert_matrix_to_dataframe_step, matrix)
