from sklearn.model_selection import GridSearchCV

from firecannon.utils import check_shape_compatibility, write_json
from firecannon.errors import IncompatibleDataShapeException


class BestParamsTestSuite:
    """
    Class to find best hyperparameters for a list of entities
    """

    def __init__(self,
                 k_folds=4,
                 n_jobs=-1,
                 verbose=False,
                 scoring='accuracy'
    ):
        self.k_folds = k_folds
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.scoring = scoring

    def __str__(self):
        return f'k_folds: {self.k_folds}\n' \
               f'n_jobs: {self.n_jobs}\n' \
               f'verbose: {self.verbose}\n' \
               f'scoring: {self.scoring}'

    def run(self, X: any, y: any, model_settings: any):
        if not check_shape_compatibility(X, y):
            raise IncompatibleDataShapeException(X.shape[0], y.shape[0])

        model = list(model_settings.keys())[0]
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=model_settings.get(model),
                                   cv=self.k_folds,
                                   n_jobs=self.n_jobs,
                                   scoring=self.scoring or model.score)
        grid_search.fit(X, y)

        return {
            grid_search.best_estimator_: grid_search.best_score_
        }


"""
    def run(self, x: any, y: any, model_parameters: dict) -> dict:
        x: any implementation of 2D matrix with features for training
        y: any implementation of 2D matrix with labels
        model_parameters: dict, which keys are instantiated entities and values are lists with hyperparameter
        if not check_shape_compatibility(x, y):
            raise IncompatibleDataShapeException(x.shape[0], y.shape[0])

        best_models = {}

        for model in model_parameters.keys():
            grid_search = GridSearchCV(estimator=model,
                                       param_grid=model_parameters.get(model),
                                       cv=self.k_folds,
                                       n_jobs=self.n_jobs,
                                       scoring=self.scoring or model.score)
            grid_search.fit(x, y)
            best_params = grid_search.best_params_

            best_models.update({model.__class__.__name__: best_params})

            if self.verbose:
                print(f'Best params for {model.__class__.__name__}: {best_params}')

        if self.output_path != 'NO_PATH':
            write_json(data=best_models, filepath=self.output_path)

        return best_models
"""