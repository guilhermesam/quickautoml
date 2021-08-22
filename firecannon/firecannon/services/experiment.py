from sklearn.linear_model import ElasticNet

from firecannon.services.runners import Classifier, Regressor
from sklearn.model_selection import cross_val_score, train_test_split
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from pmlb import fetch_data
from numpy import mean

agaricus_lepiota_X, agaricus_lepiota_y = fetch_data('agaricus_lepiota', return_X_y=True)
sonar_X, sonar_y = fetch_data('sonar', return_X_y=True)
house_16H_X, house_16H_y = fetch_data('574_house_16H', return_X_y=True)
cpu_small_X, cpu_small_y = fetch_data('529_pollen', return_X_y=True)

classification_data = {
    'agaricus_lepiota': (agaricus_lepiota_X, agaricus_lepiota_y),
    'sonar': (sonar_X, sonar_y)
}

regression_data = {
    '574_house_16H': (house_16H_X, house_16H_y),
    '529_pollen': (cpu_small_X, cpu_small_y)
}

file = open('experiment.txt', 'a')

start = time.time()

for dataset_name, data in classification_data.items():
    print(dataset_name)
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    baseline = KNeighborsClassifier()
    classifier = Classifier()
    classifier.fit(X_train, y_train)
    model_pred = classifier.predict(X_test)
    file.write(f'{dataset_name} in model: {accuracy_score(y_test, model_pred)}\n')
    file.write(f'{dataset_name} in baseline: {mean(cross_val_score(baseline, X, y, cv=4, scoring="accuracy"))}\n')

for dataset_name, data in regression_data.items():
    print(dataset_name)
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    baseline = ElasticNet(max_iter=None)
    regressor = Regressor()
    regressor.fit(X_train, y_train)
    model_pred = regressor.predict(X_test)

    file.write(f'{dataset_name} in model: {mean_squared_error(y_test, model_pred)}\n')
    file.write(f'{dataset_name} in baseline: {mean(cross_val_score(baseline, X, y, cv=4, scoring="neg_mean_squared_error"))}\n')
print('end \n')

end = time.time()
file.write(f'execução: {end - start}')
file.close()
# results = cross_validate(baseline, X, y, scoring='accuracy')

