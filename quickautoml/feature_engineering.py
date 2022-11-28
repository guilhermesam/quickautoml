from abc import abstractmethod

from pandas import Series


class FeatureEngineer:
    def __init__(self):
        super().__init__()
        self.matrix = None

    def __remove_unbalanced_columns(self):
        def get_diff_in_value_counts(value_counts: Series) -> float:
            return value_counts.min() / value_counts.sum()

        threshold: int = 10
        to_drop_rows: list[str] = [
            c for c in self.matrix.columns if get_diff_in_value_counts(self.matrix[c].value_counts()) < threshold
        ]
        self.matrix.drop(to_drop_rows, axis=1, inplace=True)
        return self

    def __count_used_permissions(self):
        permission_cols = [x for x in self.matrix.columns if 'permission' in x]
        self.matrix['permissions_used_count'] = self.matrix[self.matrix[permission_cols] == 1].count(axis=1)
        return self

    def __remove_columns_with_unique_values(self):
        to_drop_rows: list[str] = [
            c for c in self.matrix.columns if self.matrix[c].nunique() == 1
        ]
        self.matrix.drop(to_drop_rows, axis=1, inplace=True)
        return self

    def __collect(self):
        return self.matrix

    def run(self, matrix):
        self.matrix = matrix
        return self \
            .__count_used_permissions() \
            .__remove_columns_with_unique_values() \
            .__collect()
