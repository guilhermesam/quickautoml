from random import sample
from csv import writer
from os.path import exists
from time import time
from datetime import datetime
from typing import Any, List, Union, TextIO

from numpy import ndarray, array
from pandas import read_csv
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autosklearn.classification import AutoSklearnClassifier
from tpot import TPOTClassifier

from firecannon.estimators import Classifier

# TODO: Selecionar dataset e verificar se o mesmo ja existe em dataset_results.csv


class TrainingResults:
  def __init__(self,
               dataset_name: str,
               dataset_nrows: int,
               dataset_ncolumns: int,
               tpot_score: float,
               quick_score: float,
               autosklearn_score: float,
               tpot_time: float,
               quick_time: float,
               autosklearn_time: float
               ):
    self.dataset_name = dataset_name
    self.dataset_nrows = dataset_nrows
    self.dataset_ncolumns = dataset_ncolumns
    self.tpot_score = tpot_score
    self.quick_score = quick_score
    self.autosklearn_score = autosklearn_score
    self.tpot_time = tpot_time
    self.quick_time = quick_time
    self.autosklearn_time = autosklearn_time

  def to_list(self) -> List[Union[str, float]]:
    return [
      self.dataset_name, self.dataset_nrows, self.dataset_ncolumns, self.tpot_score, self.quick_score,
      self.autosklearn_score, self.tpot_time, self.quick_time, self.autosklearn_time
    ]


def run() -> None:
  SAMPLES_NUMBER = -1
  ROWS_INDEX = 0

  datasets_list = 'dataset_stats.csv'
  df = read_csv(datasets_list, sep='\t')

  if SAMPLES_NUMBER == -1:
    classification_df = df[df['task'] == 'classification']
    classification_dataset_names = classification_df['dataset']

  else:
    classification_df = df[df['task'] == 'classification']
    random_indexes = sample([x for x in range(classification_df.shape[ROWS_INDEX])], SAMPLES_NUMBER)
    classification_dataset_names = classification_df['dataset'].iloc[random_indexes]

  file = file_factory('tools_evaluation.csv')
  results_writer = csv_writer_factory(file)

  counter = 0

  write_header(results_writer)
  for dataset in classification_dataset_names:
    print(f'[INFO] Dataset of index {counter}, name: {dataset}')
    results: TrainingResults = train(dataset_name=dataset)
    write_results(results_writer, results.to_list())
    counter += 1

  close_file(file)


def train(dataset_name: str):
  quick_classifier = Classifier()
  tpot_classifier = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=4444, verbosity=0)
  autosklearn_classifier = AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30
  )
  X, y = fetch_data(dataset_name=dataset_name, return_X_y=True)
  print(f'[TPOT] Training')
  tpot_time, tpot_score = get_score_and_time(tpot_classifier, X, y)
  print(f'[QuickAutoML] Training')
  quick_time, quick_score = get_score_and_time(quick_classifier, X, y)
  print(f'[AutoSKlearn] Training')
  autosklearn_time, autosklearn_score = get_score_and_time(autosklearn_classifier, X, y)
  return TrainingResults(
    dataset_name=dataset_name,
    dataset_nrows=array(X).shape[0],
    dataset_ncolumns=array(X).shape[1],
    tpot_score=tpot_score,
    quick_score=quick_score,
    autosklearn_score=autosklearn_score,
    tpot_time=tpot_time,
    quick_time=quick_time,
    autosklearn_time=autosklearn_time
  )


def get_score_and_time(estimator: Any, X: Union[ndarray, list], y: Union[ndarray, list]):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4444)
  start = time()
  estimator.fit(X_train, y_train)
  end = time()
  predictions = estimator.predict(X_test)
  score = accuracy_score(y_test, predictions)
  return (end - start), score


def file_factory(output_path: str):
  return open(output_path, 'a')


def csv_writer_factory(file):
  return writer(file)


def write_header(csv_writer):
  csv_writer.writerow(
    ['dataset_name', 'dataset_nrows', 'dataset_ncolumns', 'tpot_score',
     'quick_score', 'autosklearn_score', 'tpot_time', 'quick_time', 'autosklearn_time']
  )


def write_results(csv_writer, results: list):
  csv_writer.writerow(results)


def close_file(file):
  file.close()


if __name__ == '__main__':
  run()
