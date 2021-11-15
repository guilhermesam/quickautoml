from time import time
from datetime import date, datetime
from typing import TextIO

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from firecannon.estimators import Classifier

file = open('logs.txt', 'a')


class TestSuite:
  def __init__(self,
               dataset_name: str,
               dataset_path: str,
               output_file: TextIO,
               ):
    self.output_file = output_file
    self.dataset_name = dataset_name
    self.dataset_path = dataset_path
    self.estimator = Classifier()

  def run(self, y_label: str = 'class'):
    start = time()
    df = read_csv(self.dataset_path)
    X = df.drop(y_label, axis=1)
    y = df[y_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    self.estimator.fit(X_train, y_train)
    predicted = self.estimator.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    end = time()
    now = datetime.now()
    self.output_file.write(f'===== RUN => {date.today()} - {now.hour}:{now.minute}:{now.second}\n')
    self.output_file.write(f'{self.dataset_name} accuracy: {accuracy}\n')
    self.output_file.write(f'{self.dataset_name} time: {end - start}\n')
    self.output_file.write(f'==============================================\n')

  def close(self):
    self.output_file.close()


androcrawl = TestSuite(
  dataset_name='Androcrawl',
  dataset_path='../datasets/androcrawl.csv',
  output_file=file
)

androcrawl.run()
androcrawl.close()
