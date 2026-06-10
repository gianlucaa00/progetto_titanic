import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from it.akron_valtellina.models.dataset_split import Split
from it.akron_valtellina.models.scaling_pca import ScalerPCA


class XGBoost:
    def __init__(self, X, y, use_pca=False, n_components=None):
        """
        XGBoost Classifier.

        Parameters:
        -----------
        X : DataFrame or array-like
            Feature matrix
        y : Series or array-like
            Target variable
        use_pca : bool, default=False
            Se True, applica PCA prima dell'addestramento
        n_components : int or None, default=None
            Numero di componenti PCA (se None e use_pca=True, seleziona automaticamente 95% varianza)
        """
        self.__X = X
        self.__y = y
        self.__model = None
        self.__results = None
        self.__use_pca = use_pca
        self.__n_components = n_components
        self.__pca_transformer = None
        self.__label_encoder = None
        self._train_columns = None

    def train(self, n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=1, gamma=0, **kwargs):
        """
        Addestra il modello XGBoost.

        Parameters:
        -----------
        n_estimators : int, default=100
            Numero di alberi
        max_depth : int, default=3
            Profondità massima degli alberi
        learning_rate : float, default=0.1
            Learning rate
        subsample : float, default=0.8
            Proporzione di campioni da usare per ogni albero
        colsample_bytree : float, default=0.8
            Proporzione di feature da usare per ogni albero
        min_child_weight : int, default=1
            Peso minimo dei figli
        gamma : float, default=0
            Riduzione minima della loss per fare uno split
        **kwargs : parametri aggiuntivi per XGBClassifier
        """

        # CORREZIONE 1: Gestione PCA vs scaling semplice
        if self.__use_pca:
            # Applica PCA
            pca_transformer = ScalerPCA(self.__X, self.__y, n_components=self.__n_components, dummies=True)
            X_train, X_test, y_train, y_test = pca_transformer.get_pca()
            self.__pca_transformer = pca_transformer

            # Stampa info sulla varianza spiegata
            try:
                pca_info = pca_transformer.get_pca_info()
                print(f"PCA: {pca_info['n_components_']} componenti selezionate")
                print(f"Varianza spiegata cumulata: {pca_info['cumulative_variance_'][-1]:.2%}")
            except:
                pass
        else:
            # Solo split con dummies
            splitter = Split(self.__X, self.__y, dummies=True)
            X_train, X_test, y_train, y_test = splitter.get_split(dummies=True)

        if hasattr(X_train, 'columns'):
            self._train_columns = X_train.columns.tolist()

        # Codifica del target se necessario
        if isinstance(y_train, pd.Series) and y_train.dtype == 'object':
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            self.__label_encoder = le
        elif isinstance(y_train, (pd.Series, np.ndarray)) and y_train.dtype.kind in ['U', 'S', 'O']:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            self.__label_encoder = le

        # Addestra il modello
        self.__model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            **kwargs
        )

        self.__model.fit(X_train, y_train)

        # Predizioni
        y_pred_train = self.__model.predict(X_train)
        y_pred_test = self.__model.predict(X_test)
        y_proba_test = self.__model.predict_proba(X_test) if hasattr(self.__model, 'predict_proba') else None

        # Feature importance
        feature_importance = None
        if hasattr(self.__model, 'feature_importances_'):
            if hasattr(X_train, 'columns'):
                feature_names = X_train.columns
            else:
                feature_names = [f'PC{i + 1}' if self.__use_pca else f'feature_{i}'
                                 for i in range(X_train.shape[1])]

            feature_importance = pd.DataFrame({
                'variabile': feature_names,
                'importanza': self.__model.feature_importances_
            }).sort_values('importanza', ascending=False)

        # Risultati completi
        self.__results = {
            'modello': self.__model,
            'feature_importance': feature_importance,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_proba_test': y_proba_test,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'use_pca': self.__use_pca,
            'params': {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'min_child_weight': min_child_weight,
                'gamma': gamma
            }
        }

        # Aggiungi info PCA se disponibile
        if self.__use_pca and hasattr(self, '__pca_transformer') and self.__pca_transformer:
            try:
                self.__results['pca_info'] = self.__pca_transformer.get_pca_info()
            except:
                pass

        return self.__results

    def _preprocess_predict_data(self, X_new):
        """
        Preprocessa i dati per la predizione usando lo stesso metodo del training.
        """
        # Caso 1: PCA è stata usata nel training
        if self.__use_pca and self.__pca_transformer is not None:
            # Usa il metodo transform_new_data del PCA transformer
            return self.__pca_transformer.transform_new_data(X_new)

        # Caso 2: Nessuna PCA, preprocessing standard con dummies
        # Applica one-hot encoding come nel training
        if hasattr(X_new, 'columns'):
            # Identifica colonne categoriche
            categorical_cols = X_new.select_dtypes(include=['object', 'string', 'category']).columns

            if len(categorical_cols) > 0:
                # Applica one-hot encoding
                dummies = pd.get_dummies(X_new[categorical_cols], drop_first=True, dtype=int)
                numeric_cols = X_new.select_dtypes(include=['int64', 'float64']).columns
                X_processed = pd.concat([X_new[numeric_cols], dummies], axis=1)
            else:
                X_processed = X_new.copy()

            # Se abbiamo le colonne del training, allinea
            if self._train_columns:
                # Aggiungi colonne mancanti con valore 0
                for col in self._train_columns:
                    if col not in X_processed.columns:
                        X_processed[col] = 0
                # Seleziona solo le colonne del training e riordina
                X_processed = X_processed[self._train_columns]

            return X_processed

        # Se X_new è un array numpy
        return X_new

    def predict(self, X_new):
        """
        Predice su nuovi dati applicando lo stesso preprocessing del training.
        """
        if self.__model is None:
            raise ValueError("Devi prima addestrare il modello con train()")

        # Preprocessa i dati
        X_processed = self._preprocess_predict_data(X_new)

        # Predici
        return self.__model.predict(X_processed)

    def predict_proba(self, X_new):
        """
        Predice probabilità su nuovi dati.
        """
        if self.__model is None:
            raise ValueError("Devi prima addestrare il modello con train()")

        if not hasattr(self.__model, 'predict_proba'):
            raise ValueError("Questo modello non supporta predict_proba")

        X_processed = self._preprocess_predict_data(X_new)
        return self.__model.predict_proba(X_processed)

    def get_results(self):
        """Restituisce i risultati dell'addestramento."""
        if self.__results is None:
            raise ValueError("Devi prima addestrare il modello con train()")
        return self.__results

    def get_model(self):
        """Restituisce il modello addestrato."""
        if self.__model is None:
            raise ValueError("Devi prima addestrare il modello con train()")
        return self.__model
