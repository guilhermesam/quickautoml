from sklearn.tree import DecisionTreeClassifier

from firecannon.services.best_hyperparams import BestParamsTestSuite
from firecannon.services.parameterized_comparation import ParameterizedTestSuite
from firecannon.presentation.colors import ConsoleColors

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


class Classifier:
    def __init__(self, test_settings: dict = None):
        self.__fitted = False
        self.test_settings = test_settings
        self.best_model = None

    def fit(self, X, y):
        rf = RandomForestClassifier()
        knn = KNeighborsClassifier()
        ada = AdaBoostClassifier()

        model_settings = {
            knn: {
                'n_neighbors': [3, 5, 7],
                'leaf_size': [15, 30, 45]
            },
            rf: {
                'n_estimators': [50, 100, 150],
                'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'log2', 'sqrt'],
            },
            ada: {
                'n_estimators': [50, 100, 150],
                'learning_rate': [1, 0.1, 0.5]
            }
        }

        print('Buscando melhores hiperparâmetros...', end='')
        best_hyperparams_test = BestParamsTestSuite(self.test_settings)
        best_models = []
        for model, params in model_settings.items():
            best_model = best_hyperparams_test.run(X, y, {model: params})
            best_models.append(best_model)

        print(f'{ConsoleColors.OKGREEN}OK{ConsoleColors.END_LINE}')

        print('Buscando o melhor modelo...', end='')
        best_models_test = ParameterizedTestSuite(
            {
                'k_folds': 3,
                'float_precision': 3,
                'problem_type': 'classification',
                'stratify': True,
                'report_type': 'csv',
                'metric': 'accuracy'
            }
        )
        scores = best_models_test.run(X, y, best_models)
        self.__fitted = True
        self.best_model = best_models_test.get_best_model(scores)
        print(f'{ConsoleColors.OKGREEN}OK{ConsoleColors.END_LINE}')

    def predict(self, y):
        return self.best_model.predict(y)


if __name__ == '__main__':
    from pandas import read_csv
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score
    from sklearn.svm import SVC

    df = read_csv('selected_features.csv')
    X = df.drop('class', axis=1)
    y = df['class']
    classifier = Classifier()
    svc_model = DecisionTreeClassifier()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)
    classifier.fit(X_train, y_train)
    svc_model.fit(X_train, y_train)

    svc_predictions = svc_model.predict(X_test)
    classifier_predictions = classifier.predict(X_test)

    print(f'SVC Accuracy: {round(precision_score(y_test, svc_predictions), 4)}')
    print(f'Classifier Accuracy: {round(precision_score(y_test, classifier_predictions), 4)}')
