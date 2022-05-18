from quickautoml.adapters import SKLearnModelsSupplier
from quickautoml.estimators import Classifier
from quickautoml.feature_engineering import PandasFeatureEngineer
from quickautoml.hyperparameter_optimizer import OptunaHyperparamsOptimizer
from quickautoml.preprocessors import PandasDataPreprocessor
from quickautoml.entities import NaiveModel, Hyperparameter

import pandas as pd


def make_classifier():
  data_preprocessor = PandasDataPreprocessor()
  feature_engineer = PandasFeatureEngineer()
  hyperparameter_optimizer = OptunaHyperparamsOptimizer('accuracy')
  models_supplier = SKLearnModelsSupplier()
  return Classifier(data_preprocessor, feature_engineer, models_supplier, hyperparameter_optimizer)
