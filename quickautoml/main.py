from quickautoml.adapters import SKLearnModelsSupplier
from quickautoml.entities import TrainingConfig
from quickautoml.estimators import Classifier
from quickautoml.services import OptunaHyperparamsOptimizer
from quickautoml.utils import generate_fake_data

if __name__ == '__main__':
  training_config = TrainingConfig()
  training_config.metric = 'accuracy'
  training_config.report_type = 'json'
  hyperparameter_optimizer = OptunaHyperparamsOptimizer(training_config.metric)
  models_supplier = SKLearnModelsSupplier()
  estimator = Classifier(training_config, hyperparameter_optimizer, models_supplier)

  x_train, x_test, y_train, y_test = generate_fake_data()
  estimator.fit(x_train, y_train)
  estimator.predict(x_test)
  print(estimator.best_model)
