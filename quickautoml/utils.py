from json import dump, load
from io import FileIO

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from quickautoml.exceptions import IncompatibleDataShapeException


def generate_fake_data():
    x, y = make_classification()
    # x_train, x_test, y_train, y_test
    return train_test_split(x, y, test_size=0.20, random_state=42)


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


def check_shape_compatibility(x: any, y: any) -> None:
    row_index = 0
    if x.shape[row_index] != y.shape[row_index]:
        raise IncompatibleDataShapeException(x.shape[0], y.shape[0])
