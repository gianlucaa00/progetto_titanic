import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from it.akron_valtellina.models.pca import ScalerPCA


class KFoldCrossValidation:
    def __init__(self, model_class, model_params=None, k=5, method='kfold',
                 use_pca=False, n_components=None, classification=True, random_state=42):
        """
        K-Fold Cross Validation per valutare modelli.

        Parameters:
        -----------
        model_class : class
            Classe del modello da validare (es. RegressioneLogistica, RandomForest, etc.)
        model_params : dict, optional
            Parametri da passare al costruttore del modello (X, y, use_pca, n_components)
        k : int, default=5
            Numero di fold per k-fold (ignorato se method='loo' o 'lho')
        method : str, default='kfold'
            Metodo di cross-validation: 'kfold', 'loo' (Leave One Out), 'lho' (Leave Half Out)
        use_pca : bool, default=False
            Se True, applica PCA nei modelli
        n_components : int or None, default=None
            Numero di componenti PCA
        classification : bool, default=True
            Se True, usa metriche di classificazione, altrimenti di regressione
        random_state : int, default=42
            Seed per riproducibilità
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.k = k
        self.method = method
        self.use_pca = use_pca
        self.n_components = n_components
        self.classification = classification
        self.random_state = random_state
        self.__results = None

    def validate(self, X, y, train_params=None):
        """
        Esegue la cross-validation sul modello.

        Parameters:
        -----------
        X : DataFrame or array-like
            Feature matrix
        y : Series or array-like
            Target variable
        train_params : dict, optional
            Parametri da passare al metodo train() del modello

        Returns:
        --------
        dict: Risultati della cross-validation
        """
        train_params = train_params or {}

        # Configura il metodo di cross-validation
        if self.method == 'loo':
            cv = LeaveOneOut()
            n_splits = len(X)

        elif self.method == 'lho':
            # Leave Half Out: usa metà dei dati per test
            n_splits = 2
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        else:  # kfold standard
            # Usa StratifiedKFold per classificazione, KFold per regressione
            if self.classification:
                cv = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=self.k, shuffle=True, random_state=self.random_state)

        # Liste per memorizzare i risultati
        scores = []
        models = []
        all_predictions = []
        all_true_values = []

        # Esegue la cross-validation
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):

            # Split dei dati
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_test_fold = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            y_test_fold = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]

            # Crea e addestra il modello
            model = self.model_class(
                X_train_fold,
                y_train_fold,
                use_pca=self.use_pca,
                n_components=self.n_components,
                **self.model_params
            )

            results = model.train(**train_params)

            # Calcola metrica appropriata
            y_pred = results['y_pred_test']
            y_true = results['y_test']

            if self.classification:
                score = accuracy_score(y_true, y_pred)
            else:
                score = r2_score(y_true, y_pred)

            scores.append(score)
            models.append(model)
            all_predictions.extend(y_pred)
            all_true_values.extend(y_true)

        # Calcola statistiche
        scores = np.array(scores)

        self.__results = {
            'scores_per_fold': scores,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max(),
            'models': models,
            'all_predictions': np.array(all_predictions),
            'all_true_values': np.array(all_true_values),
            'method': self.method,
            'k': self.k if self.method == 'kfold' else None,
            'n_folds': len(scores)
        }

        return self.__results

    def get_results(self):
        """Restituisce i risultati della cross-validation."""
        if self.__results is None:
            raise ValueError("Devi prima eseguire validate()")
        return self.__results

    def get_best_model(self):
        """Restituisce il modello con le migliori performance."""
        if self.__results is None:
            raise ValueError("Devi prima eseguire validate()")

        best_idx = np.argmax(self.__results['scores_per_fold'])
        return self.__results['models'][best_idx]


