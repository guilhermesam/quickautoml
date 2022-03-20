from quickautoml.adapters import SKLearnModelsSupplier
from quickautoml.entities import TrainingConfig
from quickautoml.estimators import Classifier
from quickautoml.services import OptunaHyperparamsOptimizer
from quickautoml.utils import generate_fake_data
from quickautoml.entities import NaiveModel, Hyperparameter


def make_classifier():
  training_config = TrainingConfig()
  hyperparameter_optimizer = OptunaHyperparamsOptimizer(training_config.metric)
  models_supplier = SKLearnModelsSupplier()
  return Classifier(training_config, hyperparameter_optimizer, models_supplier)


if __name__ == '__main__':
  estimator = make_classifier()
  x_train, x_test, y_train, y_test = generate_fake_data()

  default = {
      NaiveModel(name='KNeighbors Classifier', estimator=SKLearnModelsSupplier().get_model('knn-c')): [
        Hyperparameter(name='n_neighbors', data_type='int', min_value=3, max_value=7),
        Hyperparameter(name='leaf_size', data_type='int', min_value=15, max_value=60)
      ],
      NaiveModel(name='Adaboost Classifier', estimator=SKLearnModelsSupplier().get_model('ada-c')): [
        Hyperparameter(name='n_estimators', data_type='int', min_value=10, max_value=100),
        Hyperparameter(name='learning_rate', data_type='float', min_value=0.1, max_value=2)
      ]
    }

  estimator.training_config.search_space = default
  estimator.fit(x_train, y_train)
  estimator.predict(x_test)
  print(estimator.best_model)
