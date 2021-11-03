import numpy as np
from json import dump
from numpy import ndarray, array

from firecannon.exceptions import IncompatibleDataShapeException


def write_json(data: dict, filepath: str) -> None:
    try:
        with open(f'{filepath}.json', 'w') as file:
            dump(data, file, indent=4)
    except IOError as error:
        raise error(f'I/O Error: {error}')

def check_shape_compatibility(X: any, y: any) -> bool:
    X = convert_to_np_array(X)
    y = convert_to_np_array(y)

    ROW_INDEX = 0
    if X.shape[ROW_INDEX] != y.shape[ROW_INDEX]:
        raise IncompatibleDataShapeException(X.shape[0], y.shape[0])


def convert_to_np_array(vector: any) -> ndarray:
    if not isinstance(vector, ndarray):
        return array(vector)
    return vector


def get_problem_type(vector: any) -> str:
    vector = convert_to_np_array(vector)
    if vector.dtype == np.float:
        return 'regression'
    else:
        return 'classification'
