from random import sample
from csv import writer
from os.path import exists
from time import time
from datetime import datetime
from typing import Any, List, Union, TextIO

from numpy import ndarray
from pandas import read_csv
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tpot import TPOTClassifier
from firecannon.estimators import Classifier


class TrainingResults:
  def __init__(self,
               dataset_name: str,
               tpot_score: float,
               quick_score: float,
               tpot_time: float,
               quick_time: float
               ):
    self.dataset_name = dataset_name
    self.tpot_score = tpot_score
    self.quick_score = quick_score
    self.tpot_time = tpot_time
    self.quick_time = quick_time

  def to_list(self) -> List[Union[str, float]]:
    return [
      self.dataset_name, self.tpot_score, self.quick_score, self.tpot_time, self.quick_time
    ]


def run() -> None:
  SAMPLES_NUMBER = 5
  ROWS_INDEX = 0

  datasets_list = '../datasets/dataset_stats.tsv'
  df = read_csv(datasets_list, sep='\t')

  classification_df = df[df['task'] == 'classification']
  random_indexes = sample([x for x in range(classification_df.shape[ROWS_INDEX])], SAMPLES_NUMBER)
  random_samples_df = classification_df['dataset'].iloc[random_indexes]

  results_writer = csv_writer_factory('datasets_results.csv')
  write_header(results_writer)

  for dataset in random_samples_df:
    results: TrainingResults = train(dataset_name=dataset)
    write_results(results_writer, results.to_list())


def train(dataset_name: str):
  quick_classifier = Classifier()
  tpot_classifier = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=4444, verbosity=2)
  X, y = fetch_data(dataset_name=dataset_name, return_X_y=True)
  quick_time, quick_score = get_score_and_time(quick_classifier, X, y)
  print(f'[INFO] {dataset_name} Dataset')
  tpot_time, tpot_score = get_score_and_time(tpot_classifier, X, y)
  return TrainingResults(
    dataset_name=dataset_name,
    tpot_score=tpot_score,
    quick_score=quick_score,
    tpot_time=tpot_time,
    quick_time=quick_time
  )


def get_score_and_time(estimator: Any, X: Union[ndarray, list], y: Union[ndarray, list]):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4444)
  start = time()
  estimator.fit(X_train, y_train)
  end = time()
  predictions = estimator.predict(X_test)
  score = accuracy_score(y_test, predictions)
  return (end - start), score


def csv_writer_factory(output_path: str):
  file = open(output_path, 'a')
  return writer(file)


def write_header(csv_writer):
  csv_writer.writerow(
    ['dataset_name', 'tpot_score', 'quick_score', 'tpot_time', 'quick_time']
  )


def write_results(csv_writer, results: list):
  csv_writer.writerow(results)


def close_file(file: TextIO):
  file.close()


if __name__ == '__main__':
  run()
