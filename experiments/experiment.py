from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pmlb import fetch_data
from firecannon.services.runners import Classifier
from tpot import TPOTClassifier
from numpy import mean

import time
import csv

# magic_X, magic_y = fetch_data('magic', return_X_y=True)
new_thyroid_X, new_thyroid_y = fetch_data('new_thyroid', return_X_y=True)
hypothyroid_X, hypothyroid_y = fetch_data('hypothyroid', return_X_y=True)
dermatology_X, dermatology_y = fetch_data('dermatology', return_X_y=True)
hayes_roth_X, hayes_roth_y = fetch_data('hayes_roth', return_X_y=True)
dis_X, dis_y = fetch_data('dis', return_X_y=True)
penguins_X, penguins_y = fetch_data('penguins', return_X_y=True)

classification_data = (
    (new_thyroid_X, new_thyroid_y),
    (hypothyroid_X, hypothyroid_y),
    (dermatology_X, dermatology_y),
    (hayes_roth_X, hayes_roth_y),
    (dis_X, dis_y),
    (penguins_X, penguins_y)
)

file = open('new_experiment.csv', 'w')
writer = csv.writer(file)
writer.writerow(['dataset', 'fire_time', 'fire_acc', 'tpot_time', 'tpot_acc'])

index = ['new_thyroid', 'hypothyroid', 'dermatology', 'hayes_roth', 'dis', 'penguins']
iterator = 0
RANDOM_STATE = 777

for data in classification_data:
    tpot_time_agg = []
    tpot_accuracy_agg = []
    firecannon_time_agg = []
    firecannon_accuracy_agg = []
    for step in range(5):
        print(f'Started {index[iterator]}, step {step}')
        classifier = Classifier()
        tpot_model = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=RANDOM_STATE, verbosity=0)
        X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.20, random_state=RANDOM_STATE)

        fire_start = time.time()
        classifier.fit(X_train, y_train)
        fire_end = time.time()

        tpot_start = time.time()
        tpot_model.fit(X_train, y_train)
        tpot_end = time.time()

        classifier_predictions = classifier.predict(X_test)
        tpot_predictions = tpot_model.predict(X_test)

        tpot_time_agg.append(tpot_end - tpot_start)
        tpot_accuracy_agg.append(round(accuracy_score(y_test, tpot_predictions), 4))
        firecannon_time_agg.append(fire_end - fire_start)
        firecannon_accuracy_agg.append(round(accuracy_score(y_test, classifier_predictions), 4))

    writer.writerow([index[iterator],
                     mean(firecannon_time_agg),
                     mean(firecannon_accuracy_agg),
                     mean(tpot_time_agg),
                     mean(tpot_accuracy_agg)
                     ])

    print(f'Ended {index[iterator]}')
    iterator += 1

file.close()
