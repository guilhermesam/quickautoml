from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
import numpy as np
from pandas import DataFrame, Series


class TestSuite():

  def __init__(self):
    pass    
   

  def check_shape_compatibility(self, X, y):
      if not isinstance(y, np.ndarray):
        y = np.array(y)
        
      ROW_INDEX = 0
      return X.shape[0] == y.shape[0]


  def write_json(self, data: dict, filepath: str):
    import json
    try:
      with open(f'{filepath}.json', 'w') as file:
        json.dump(data, file, indent=4)
    except IOError as e:
      print(f'I/O Error: {e}')
      

class BestParamsTestSuite(TestSuite):
  """
  Class to find best hyperparameters for a model
  verbose: if True, prints best_params for each model
  """

  def __init__(self, verbose=False, output_file=False):
    super(TestSuite, self).__init__()
    self.verbose = verbose
    self.output_file = output_file
  
  def run(self, X, y, model_parameters: dict, test_parameters: dict):
    """
    X: numpy array or pandas Dataframe with features for training
    y: numpy array, pandas Series or DataFrame with labels for features in X
    model_parameters: dict, which keys are instantiated sklearn models and values are lists with hyperparameter options. Ex:
      SomeModelRegressor(): {
        'hyperparameter_1': [1, 2, 3, 4],
        'hyperparameter_2': ['value', 'another_value']
      }
    test_parameters: parameters for test execution.
      {
        cv: number of folds. Default: 5,
        n_jobs: Number of jobs to run in parallel. Default: None
      }
    """
    if not self.check_shape_compatibility(X, y):
          raise ValueError('X e y possuem valores inconsistentes de amostras!')

    best_models = {}

    for model in model_parameters.keys():
      grid_search = GridSearchCV(estimator=model,
                          param_grid=model_parameters.get(model),
                          cv=test_parameters.get('cv') or 5,
                          n_jobs=test_parameters.get('n_jobs') or None)
      grid_search.fit(X, y)
      best_params = grid_search.best_params_
      
      best_models.update({model.__class__.__name__: best_params})

      if self.verbose:
        print(f'Best params for {model.__class__.__name__}: {best_params}')

    if self.output_file:
      self.write_json(data=best_models, filepath='best_models')

    return best_models


class ParameterizedTestSuite(TestSuite):

  def __init__(self, stratify=False):
    super(TestSuite, self).__init__()
    self.stratify = stratify
  

  def __make_folds(self, stratify, n_splits):
    if stratify:
      return StratifiedKFold(n_splits=n_splits)
    else:
      return KFold(n_splits=n_splits)


  def run(self, X, y, models: list, test_parameters: dict) -> DataFrame :
      if not self.check_shape_compatibility(X, y):
          raise ValueError('X e y possuem valores inconsistentes de amostras!')
      
      scores_df = DataFrame(columns=['mean', 'std'])
      kfold = self.__make_folds(self.stratify, n_splits=test_parameters.get('n_splits'))    

      for model in models:
          scores = []
          
          for train, test in kfold.split(X, y):
              model.fit(X.iloc[train], y[train])
              scores.append(model.score(X.iloc[test], y[test]))

          score_series = Series({
            'mean': np.round(np.mean(scores), test_parameters.get('float_precision')),
            'std': np.round(np.std(scores), test_parameters.get('float_precision'))
          }, name=model.__class__.__name__)

          scores_df = scores_df.append(score_series)
      
      return scores_df 


