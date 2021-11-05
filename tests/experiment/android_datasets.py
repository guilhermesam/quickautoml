from time import time

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from firecannon.estimators import Classifier

start = time()
drebin_df = read_csv('../datasets/drebin-215.csv')
drebin_model = Classifier()
X = drebin_df.drop('class', axis=1)
y = drebin_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
drebin_model.fit(X_train, y_train)
predicted = drebin_model.predict(X_test)
print(f'Accuracy in Drebin-215: {accuracy_score(y_test, predicted)}')
print(f'Drebin-215 execution time: {time() - start}', end='\n\n')

start = time()
defense_df = read_csv('../datasets/defense-droid.csv')
defense_model = Classifier()
X = defense_df.drop('class', axis=1)
y = defense_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
defense_model.fit(X_train, y_train)
predicted = defense_model.predict(X_test)
print(f'Accuracy in Defense Droid: {accuracy_score(y_test, predicted)}')
print(f'Defense droid execution time: {time() - start}')
