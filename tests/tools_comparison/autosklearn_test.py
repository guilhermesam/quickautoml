from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import sklearn.metrics

import autosklearn.classification

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_classification_example_tmp'
)

print('Start fitting')
automl.fit(X_train, y_train, dataset_name='breast_cancer')
print('End fitting')

predictions = automl.predict(X_test)
print("Accuracy score:", accuracy_score(y_test, predictions))
