from typing import Union

from numpy import ndarray, array
from pandas import DataFrame


class DataPreprocessorWrapper:
  def __init__(self, matrix: Union[DataFrame, ndarray]):
    self.matrix = matrix

  def convert_to_dataframe(self):
    if not isinstance(self.matrix, DataFrame):
      self.matrix = DataFrame(self.matrix)
    return self

  def convert_df_to_np_array(self):
    self.matrix = array(self.matrix)
    return self

  def remove_null_values(self):
    self.matrix = self.matrix.dropna(axis=1, how='all').dropna()
    return self

  def remove_duplicates(self):
    self.matrix = self.matrix.drop_duplicates()
    return self
  
  def collect(self):
    return self.matrix


class DataPreprocessor:
  @staticmethod
  def run(matrix: Union[DataFrame, ndarray]) -> ndarray:
    return DataPreprocessorWrapper(matrix)\
                                  .convert_to_dataframe()\
                                  .remove_duplicates()\
                                  .remove_null_values()\
                                  .convert_df_to_np_array() \
                                  .collect()
