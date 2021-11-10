from time import time
from datetime import date, datetime

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from firecannon.estimators import Classifier

file = open('results.txt', 'w')

amn_start = time()
drebin_df = read_csv('../datasets/android-malware-normal.csv')
drebin_model = Classifier()
X = drebin_df.drop('class', axis=1)
y = drebin_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
drebin_model.fit(X_train, y_train)
predicted = drebin_model.predict(X_test)
amn_accuracy = accuracy_score(y_test, predicted)
amn_end = time()
print(f'Accuracy in Drebin-215: {amn_accuracy}')
print(f'Drebin-215 execution time: {amn_end - amn_start}')
print(drebin_df.shape, end='\n\n')

androcrawl_start = time()
defense_df = read_csv('../datasets/androcrawl.csv')
defense_model = Classifier()
X = defense_df.drop('class', axis=1)
y = defense_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
defense_model.fit(X_train, y_train)
predicted = defense_model.predict(X_test)
androcrawl_accuracy = accuracy_score(y_test, predicted)
androcrawl_end = time()
print(f'Accuracy in Defense Droid: {androcrawl_accuracy}')
print(f'Defense droid execution time: {androcrawl_end - androcrawl_start}')
print(defense_df.shape)

now = datetime.now()
file.write(f'===== RUN => {date.today()} - {now.hour}:{now.minute}:{now.second}\n')
file.write(f'Android Malware Normal accuracy: {amn_accuracy}\n')
file.write(f'Android Malware Normal time: {amn_end - amn_start}\n')
file.write(f'Androcrawl accuracy: {androcrawl_accuracy}\n')
file.write(f'Androcrawl time: {androcrawl_end - androcrawl_start}\n')
# file.write('Comments\n')
file.write(f'==============================================\n')
file.close()
