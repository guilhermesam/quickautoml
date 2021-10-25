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

    def run(self, X: any, y: any, model: any, model_settings: dict):
        if not check_shape_compatibility(X, y):
            raise IncompatibleDataShapeException(X.shape[0], y.shape[0])

        grid_search = GridSearchCV(estimator=model,
                                   param_grid=model_settings,
                                   cv=self.k_folds,
                                   n_jobs=self.n_jobs,
                                   scoring=self.scoring or model.score)
        grid_search.fit(X, y)

        return {
            grid_search.best_estimator_: grid_search.best_score_
        }
