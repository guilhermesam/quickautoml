from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from firecannon.estimators import Classifier

androcrawl_df = read_csv('../datasets/androcrawl.csv')
androcrawl_model = Classifier()
X = androcrawl_df.drop('class', axis=1)
y = androcrawl_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
androcrawl_model.fit(X_train, y_train)
predicted = androcrawl_model.predict(X_test)
print(f'Accuracy in Androcrawl: {accuracy_score(y_test, predicted)}')

defense_df = read_csv('../datasets/defense-droid.csv')
defense_model = Classifier()
X = defense_df.drop('class', axis=1)
y = defense_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
defense_model.fit(X_train, y_train)
predicted = defense_model.predict(X_test)
print(f'Accuracy in Defense Droid: {accuracy_score(y_test, predicted)}')
