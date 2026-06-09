import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from it.akron_valtellina.models.dataset_split import Split
from it.akron_valtellina.models.scaling_pca import ScalerPCA


class RandomForest:
    def __init__(self, X, y, use_pca=False, n_components=None):
        """
        Random Forest Classifier.

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

    def train(self, n_estimators=100, max_depth=None, min_samples_split=2,
              min_samples_leaf=1, max_features='sqrt', **kwargs):
        """
        Addestra il modello Random Forest.

        Parameters:
        -----------
        n_estimators : int, default=100
            Numero di alberi nella foresta
        max_depth : int or None, default=None
            Profondità massima degli alberi
        min_samples_split : int, default=2
            Numero minimo di campioni per split
        min_samples_leaf : int, default=1
            Numero minimo di campioni per foglia
        max_features : str or int, default='sqrt'
            Numero di feature da considerare per lo split
        **kwargs : parametri aggiuntivi per RandomForestClassifier
        """

        # Gestione PCA vs split semplice
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
        self.__model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
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
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': max_features
            }
        }

        # Aggiungi info PCA se disponibile
        if self.__use_pca and hasattr(self, '__pca_transformer') and self.__pca_transformer:
            try:
                self.__results['pca_info'] = self.__pca_transformer.get_pca_info()
            except:
                pass

        return self.__results

    def predict(self, X_new):
        """
        Predice su nuovi dati.
        """
        if self.__model is None:
            raise ValueError("Devi prima addestrare il modello con train()")

        # Applica le stesse trasformazioni
        if self.__use_pca and self.__pca_transformer is not None:
            X_transformed = self.__pca_transformer.transform_new_data(X_new)
        else:
            # Applica lo stesso split e scaling
            splitter = Split(X_new, None, dummies=True)
            X_transformed, _, _, _ = splitter.get_split(dummies=True)

        predictions = self.__model.predict(X_transformed)

        # Decodifica le etichette se necessario
        if self.__label_encoder is not None:
            predictions = self.__label_encoder.inverse_transform(predictions)

        return predictions

    def predict_proba(self, X_new):
        """
        Predice probabilità su nuovi dati.
        """
        if self.__model is None:
            raise ValueError("Devi prima addestrare il modello con train()")

        if not hasattr(self.__model, 'predict_proba'):
            raise ValueError("Questo modello non supporta predict_proba")

        # Applica le stesse trasformazioni
        if self.__use_pca and self.__pca_transformer is not None:
            X_transformed = self.__pca_transformer.transform_new_data(X_new)
        else:
            splitter = Split(X_new, None, dummies=True)
            X_transformed, _, _, _ = splitter.get_split(dummies=True)

        return self.__model.predict_proba(X_transformed)

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


