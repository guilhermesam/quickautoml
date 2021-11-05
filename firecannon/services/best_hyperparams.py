from sklearn.model_selection import GridSearchCV

from firecannon.entities import Model


class BestParamsTestSuite:
    """
    Class to find best hyperparameters for a list of entities
    """

    def __init__(self,
                 k_folds,
                 n_jobs,
                 verbose,
                 scoring
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
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=model_settings,
                                   cv=self.k_folds,
                                   verbose=self.verbose,
                                   n_jobs=self.n_jobs,
                                   scoring=self.scoring or model.score)
        grid_search.fit(X, y)

        return Model(
          name=grid_search.best_estimator_.__str__(),
          score=grid_search.best_score_,
          estimator=grid_search.best_estimator_
        )
