from csv import writer
from time import time
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
  datasets_list = 'datasets_stats.csv'
  df = read_csv(datasets_list, sep=',')

  classification_df = df[df['task'] == 'classification']
  classification_dataset_names = classification_df['dataset']

  file = open('tools_evaluation.csv', 'a')
  results_writer = writer(file)

  counter = 0

  results_writer.writerow(
    ['dataset_name', 'dataset_nrows', 'dataset_ncolumns', 'tpot_score',
     'quick_score', 'autosklearn_score', 'tpot_time', 'quick_time', 'autosklearn_time']
  )
  for dataset in classification_dataset_names:
    try:
      print(f'[INFO] Dataset of index {counter}, name: {dataset}')
      results: TrainingResults = train(dataset_name=dataset)
      results_writer.writerow(results.to_list())
      counter += 1
    except Exception as e:
      print(e)

  file.close()


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


if __name__ == '__main__':
  run()
