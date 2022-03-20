from abc import abstractmethod
from typing import List

from quickautoml.entities import FeatureEngineer

from pandas import DataFrame, Series


class PandasFeatureEngineer(FeatureEngineer):
  def __init__(self):
    super().__init__()

  def remove_umbalanced_columns(self, matrix: DataFrame):
    def get_diff_in_value_counts(value_counts: Series) -> float:
      return value_counts.min() / value_counts.sum()
    threshold: int = 10
    to_drop_rows: List[str] = [
      c for c in matrix.columns if get_diff_in_value_counts(matrix[c].value_counts()) < threshold
    ]
    return matrix.drop(to_drop_rows, axis=1)

  def count_used_permissions(self, matrix: DataFrame) -> None:
    permission_cols = [x for x in matrix.columns if 'permission' in x]
    matrix['permissions_used_count'] = matrix[[matrix[permission_cols]] == 1].count(axis=1)

  def remove_columns_with_unique_values(self, matrix: DataFrame):
    to_drop_rows: List[str] = [
      c for c in matrix.columns if matrix[c].nunique() == 1
    ]
    return matrix.drop(to_drop_rows, axis=1)
