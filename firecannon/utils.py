import numpy as np
from json import dump, load
from numpy import ndarray, array


def write_json(data: dict, filepath: str) -> None:
    try:
        with open(f'{filepath}.json', 'w') as file:
            dump(data, file, indent=4)
    except IOError as error:
        raise error(f'I/O Error: {error}')


def check_shape_compatibility(x: any, y: any) -> bool:
    x = convert_to_np_array(x)
    y = convert_to_np_array(y)

    ROW_INDEX = 0
    return x.shape[ROW_INDEX] == y.shape[ROW_INDEX]


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
