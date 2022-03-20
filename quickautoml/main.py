from quickautoml.adapters import SKLearnModelsSupplier
from quickautoml.estimators import Classifier
from quickautoml.feature_engineering import PandasFeatureEngineer
from quickautoml.hyperparameter_optimizer import OptunaHyperparamsOptimizer
from quickautoml.preprocessors import PandasDataPreprocessor
from quickautoml.utils import generate_fake_data
from quickautoml.entities import NaiveModel, Hyperparameter


def make_classifier():
  data_preprocessor = PandasDataPreprocessor()
  feature_engineer = PandasFeatureEngineer()
  hyperparameter_optimizer = OptunaHyperparamsOptimizer('accuracy')
  models_supplier = SKLearnModelsSupplier()
  return Classifier(data_preprocessor, feature_engineer, models_supplier, hyperparameter_optimizer)


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
