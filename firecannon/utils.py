from json import dump, load
from io import FileIO

from firecannon.exceptions import IncompatibleDataShapeException


def load_json(path: str) -> dict:
    file = open(path)
    data = load(file)
    file.close()
    return data


def write_json(data: dict, filepath: str) -> None:
    try:
        with open(f'{filepath}.json', 'w') as file:
            dump(data, file, indent=4)
    except IOError as error:
        raise error(f'I/O Error: {error}')


def close_file(file: FileIO):
    file.close()


def check_shape_compatibility(X: any, y: any) -> None:
    ROW_INDEX = 0
    if X.shape[ROW_INDEX] != y.shape[ROW_INDEX]:
        raise IncompatibleDataShapeException(X.shape[0], y.shape[0])
