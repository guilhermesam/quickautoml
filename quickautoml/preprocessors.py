from typing import Union

from numpy import ndarray, array
from pandas import DataFrame

from quickautoml.entities import DataPreprocessor


class PandasDataPreprocessor(DataPreprocessor):
  def __init__(self):
    super().__init__()

  def __convert_to_dataframe(self):
    if not isinstance(self.matrix, DataFrame):
      self.matrix = DataFrame(self.matrix)
    return self

  def __convert_df_to_np_array(self):
    self.matrix = array(self.matrix)
    return self

  def __remove_null_values(self):
    self.matrix = self.matrix.dropna(axis=1, how='all').dropna()
    return self

  def __remove_duplicates(self):
    self.matrix = self.matrix.drop_duplicates()
    return self
  
  def __collect(self):
    return self.matrix

  def run(self, matrix: Union[DataFrame, ndarray]):
    self.matrix = matrix
    return self.__convert_to_dataframe()\
               .__remove_duplicates()\
               .__remove_null_values()\
               .__convert_df_to_np_array() \
               .__collect()
