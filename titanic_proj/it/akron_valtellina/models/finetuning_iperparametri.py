from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, r2_score, make_scorer


class FineTuning:
    def __init__(self, model_class, model_init_params=None, classification=True,
                 cv=5, random_state=42, n_iter_random=20):
        """
        Fine-tuning di iperparametri con Grid Search e Random Search.

        Parameters:
        -----------
        model_class : class
            Classe del modello da ottimizzare
        model_init_params : dict, optional
            Parametri fissi per il costruttore del modello
        classification : bool, default=True
            Se True, usa metriche di classificazione
        cv : int, default=5
            Numero di fold per cross-validation interna
        random_state : int, default=42
            Seed per riproducibilità
        n_iter_random : int, default=20
            Numero di iterazioni per Random Search
        """
        self.model_class = model_class
        self.model_init_params = model_init_params or {}
        self.classification = classification
        self.cv = cv
        self.random_state = random_state
        self.n_iter_random = n_iter_random
        self.__best_params = None
        self.__cv_results = None
        self.__search_type = None
        self.__best_score_value = None

    def grid_search(self, X, y, param_grid, scoring=None, train_params=None):
        """
        Esegue Grid Search per ottimizzare gli iperparametri.

        Returns:
        --------
        dict: Migliori parametri e risultati in formato JSON
        """
        self.__search_type = 'grid_search'

        # Configura la metrica di scoring
        if scoring is None:
            scoring = 'accuracy' if self.classification else 'r2'

        # Configura la cross-validation
        if self.classification:
            cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        # Crea un wrapper per il modello
        class ModelWrapper:
            def __init__(self, model_class, init_params, train_params):
                self.model_class = model_class
                self.init_params = init_params
                self.train_params = train_params
                self.model = None

            def fit(self, X, y):
                self.model = self.model_class(X, y, **self.init_params)
                self.model.train(**self.train_params)
                return self

            def predict(self, X):
                return self.model.predict(X)

            def score(self, X, y):
                pred = self.predict(X)
                return accuracy_score(y, pred) if self.init_params.get('classification', True) else r2_score(y, pred)

        wrapper = ModelWrapper(self.model_class, self.model_init_params, train_params or {})

        # Esegue Grid Search
        grid_search = GridSearchCV(
            estimator=wrapper,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
            return_train_score=True
        )

        grid_search.fit(X, y)

        self.__best_params = grid_search.best_params_
        self.__best_score_value = float(grid_search.best_score_)

        # Prepara i risultati in formato JSON
        self.__cv_results = {
            'search_type': 'grid_search',
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'scoring_metric': scoring,
            'cv_folds': self.cv,
            'n_candidates': len(grid_search.cv_results_['params']),
            'all_params_tested': [str(p) for p in grid_search.cv_results_['params']],
            'mean_test_scores': [float(s) for s in grid_search.cv_results_['mean_test_score']],
            'std_test_scores': [float(s) for s in grid_search.cv_results_['std_test_score']],
            'rank_test_scores': [int(r) for r in grid_search.cv_results_['rank_test_score']]
        }

        return self.__cv_results

    def random_search(self, X, y, param_distributions, scoring=None, train_params=None, n_iter=None):
        """
        Esegue Random Search per ottimizzare gli iperparametri.

        Returns:
        --------
        dict: Migliori parametri e risultati in formato JSON
        """
        self.__search_type = 'random_search'
        n_iter = n_iter or self.n_iter_random

        # Configura la metrica di scoring
        if scoring is None:
            scoring = 'accuracy' if self.classification else 'r2'

        # Configura la cross-validation
        if self.classification:
            cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        # Crea un wrapper per il modello
        class ModelWrapper:
            def __init__(self, model_class, init_params, train_params):
                self.model_class = model_class
                self.init_params = init_params
                self.train_params = train_params
                self.model = None

            def fit(self, X, y):
                self.model = self.model_class(X, y, **self.init_params)
                self.model.train(**self.train_params)
                return self

            def predict(self, X):
                return self.model.predict(X)

            def score(self, X, y):
                pred = self.predict(X)
                return accuracy_score(y, pred) if self.init_params.get('classification', True) else r2_score(y, pred)

        wrapper = ModelWrapper(self.model_class, self.model_init_params, train_params or {})

        # Esegue Random Search
        random_search = RandomizedSearchCV(
            estimator=wrapper,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
            random_state=self.random_state,
            return_train_score=True
        )

        random_search.fit(X, y)

        self.__best_params = random_search.best_params_
        self.__best_score_value = float(random_search.best_score_)

        # Prepara i risultati in formato JSON
        self.__cv_results = {
            'search_type': 'random_search',
            'n_iterations': n_iter,
            'best_params': random_search.best_params_,
            'best_score': float(random_search.best_score_),
            'scoring_metric': scoring,
            'cv_folds': self.cv,
            'n_candidates': len(random_search.cv_results_['params']),
            'mean_test_scores': [float(s) for s in random_search.cv_results_['mean_test_score']],
            'std_test_scores': [float(s) for s in random_search.cv_results_['std_test_score']]
        }

        return self.__cv_results

    def get_best_params(self):
        """Restituisce i migliori parametri trovati."""
        if self.__best_params is None:
            raise ValueError("Devi prima eseguire grid_search() o random_search()")
        return self.__best_params

    def get_best_score(self):
        """Restituisce il miglior score trovato."""
        if self.__best_score_value is None:
            raise ValueError("Devi prima eseguire grid_search() o random_search()")
        return self.__best_score_value

    def get_results(self):
        """Restituisce tutti i risultati della ricerca."""
        if self.__cv_results is None:
            raise ValueError("Devi prima eseguire grid_search() o random_search()")
        return self.__cv_results