import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


class FineTuning:
    def __init__(self, model_class, model_init_params=None, classification=True,
                 cv=5, random_state=42, n_iter_random=20):
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

    def _model_name(self):
        return getattr(self.model_class, '__name__', str(self.model_class)).lower()

    def _one_hot_encoder(self):
        try:
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown='ignore', sparse=False)

    def _build_preprocessor(self, X):
        if not isinstance(X, pd.DataFrame):
            return 'passthrough'

        numeric_cols = X.select_dtypes(include=['number', 'bool']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

        transformers = []
        if numeric_cols:
            transformers.append((
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]),
                numeric_cols
            ))
        if categorical_cols:
            transformers.append((
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', self._one_hot_encoder())
                ]),
                categorical_cols
            ))

        if not transformers:
            return 'passthrough'

        return ColumnTransformer(transformers=transformers, remainder='drop')

    def _build_model(self):
        name = self._model_name()

        if 'regressionelogistica' in name or 'logistic' in name:
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                **self.model_init_params
            )

        if 'supportvectormachine' in name or 'svc' in name or 'svm' in name:
            return SVC(
                probability=True,
                random_state=self.random_state,
                **self.model_init_params
            )

        if 'randomforest' in name:
            return RandomForestClassifier(
                random_state=self.random_state,
                **self.model_init_params
            )

        if 'xgboost' in name or 'xgb' in name:
            return XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                **self.model_init_params
            )

        raise ValueError(f"Classe modello non supportata per fine tuning: {self.model_class}")

    def _build_estimator(self, X):
        return Pipeline([
            ('preprocess', self._build_preprocessor(X)),
            ('model', self._build_model())
        ])

    def _normalize_search_space(self, params):
        normalized = {}
        for key, value in (params or {}).items():
            normalized[key if key.startswith('model__') else f'model__{key}'] = value
        return normalized

    def _default_random_space(self):
        name = self._model_name()

        if 'regressionelogistica' in name or 'logistic' in name:
            return {
                'model__C': uniform(0.01, 10.0),
                'model__solver': ['lbfgs', 'liblinear'],
                'model__max_iter': [1000, 2000]
            }

        if 'supportvectormachine' in name or 'svc' in name or 'svm' in name:
            return {
                'model__C': uniform(0.1, 10.0),
                'model__gamma': ['scale', 'auto'],
                'model__kernel': ['rbf', 'linear']
            }

        if 'randomforest' in name:
            return {
                'model__n_estimators': randint(50, 250),
                'model__max_depth': [None, 3, 5, 8, 12],
                'model__min_samples_split': randint(2, 10),
                'model__min_samples_leaf': randint(1, 5)
            }

        if 'xgboost' in name or 'xgb' in name:
            return {
                'model__n_estimators': randint(50, 250),
                'model__max_depth': randint(2, 8),
                'model__learning_rate': uniform(0.01, 0.29),
                'model__subsample': uniform(0.6, 0.4),
                'model__colsample_bytree': uniform(0.6, 0.4)
            }

        return {}

    def _cv(self, y):
        if self.classification:
            return StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

    def _clean_best_params(self, params):
        return {
            key.replace('model__', ''): value
            for key, value in params.items()
        }

    def grid_search(self, X, y, param_grid, scoring=None, train_params=None):
        self.__search_type = 'grid_search'
        scoring = scoring or ('accuracy' if self.classification else 'r2')
        search_space = self._normalize_search_space(param_grid)

        if not search_space:
            raise ValueError("param_grid non puo essere vuoto per Grid Search")

        search = GridSearchCV(
            estimator=self._build_estimator(X),
            param_grid=search_space,
            cv=self._cv(y),
            scoring=scoring,
            n_jobs=1,
            return_train_score=True,
            error_score='raise'
        )
        search.fit(X, y)

        self.__best_params = self._clean_best_params(search.best_params_)
        self.__best_score_value = float(search.best_score_)
        self.__cv_results = {
            'search_type': 'grid_search',
            'best_params': self.__best_params,
            'best_score': self.__best_score_value,
            'scoring_metric': scoring,
            'cv_folds': self.cv,
            'n_candidates': len(search.cv_results_['params']),
            'mean_test_scores': [float(s) for s in search.cv_results_['mean_test_score']],
            'std_test_scores': [float(s) for s in search.cv_results_['std_test_score']],
            'rank_test_scores': [int(r) for r in search.cv_results_['rank_test_score']]
        }
        return self.__cv_results

    def random_search(self, X, y, param_distributions, scoring=None, train_params=None, n_iter=None):
        self.__search_type = 'random_search'
        scoring = scoring or ('accuracy' if self.classification else 'r2')
        n_iter = n_iter or self.n_iter_random
        search_space = self._normalize_search_space(param_distributions) or self._default_random_space()

        if not search_space:
            raise ValueError("param_distributions non puo essere vuoto per Random Search")

        search = RandomizedSearchCV(
            estimator=self._build_estimator(X),
            param_distributions=search_space,
            n_iter=n_iter,
            cv=self._cv(y),
            scoring=scoring,
            n_jobs=1,
            random_state=self.random_state,
            return_train_score=True,
            error_score='raise'
        )
        search.fit(X, y)

        self.__best_params = self._clean_best_params(search.best_params_)
        self.__best_score_value = float(search.best_score_)
        self.__cv_results = {
            'search_type': 'random_search',
            'n_iterations': n_iter,
            'best_params': self.__best_params,
            'best_score': self.__best_score_value,
            'scoring_metric': scoring,
            'cv_folds': self.cv,
            'n_candidates': len(search.cv_results_['params']),
            'mean_test_scores': [float(s) for s in search.cv_results_['mean_test_score']],
            'std_test_scores': [float(s) for s in search.cv_results_['std_test_score']]
        }
        return self.__cv_results

    def get_best_params(self):
        if self.__best_params is None:
            raise ValueError("Devi prima eseguire grid_search() o random_search()")
        return self.__best_params

    def get_best_score(self):
        if self.__best_score_value is None:
            raise ValueError("Devi prima eseguire grid_search() o random_search()")
        return self.__best_score_value

    def get_results(self):
        if self.__cv_results is None:
            raise ValueError("Devi prima eseguire grid_search() o random_search()")
        return self.__cv_results
