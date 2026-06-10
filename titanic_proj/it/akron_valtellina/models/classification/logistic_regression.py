import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from io import BytesIO
from it.akron_valtellina.models.dataset_split import Split
from it.akron_valtellina.models.scaling_pca import ScalerPCA


class RegressioneLogistica:
    def __init__(self, X, y, use_pca=False, n_components=None):
        """
        Logistic Regression Classifier.

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
        self._scaler = None  # Per salvare lo scaler se usato

    def train(self, C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, class_weight=None, **kwargs):
        """
        Addestra il modello di regressione logistica.
        """
        # Gestione PCA vs splitting semplice
        if self.__use_pca:
            # Applica PCA
            pca_transformer = ScalerPCA(self.__X, self.__y, n_components=self.__n_components, dummies=True)
            X_train, X_test, y_train, y_test = pca_transformer.get_pca()
            self.__pca_transformer = pca_transformer
            self._scaler = pca_transformer  # Salva il trasformatore

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

        # Salva le colonne del training per future predizioni
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

        # Gestione solvers per diversi penalty
        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            print(f"Attenzione: penalty='l1' richiede solver='liblinear' o 'saga'. Uso solver='liblinear'")
            solver = 'liblinear'
        elif penalty == 'elasticnet' and solver != 'saga':
            print(f"Attenzione: penalty='elasticnet' richiede solver='saga'. Uso solver='saga'")
            solver = 'saga'

        # Addestra il modello
        self.__model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=42,
            **kwargs
        )

        self.__model.fit(X_train, y_train)

        # Predizioni
        y_pred_train = self.__model.predict(X_train)
        y_pred_test = self.__model.predict(X_test)
        y_proba_test = self.__model.predict_proba(X_test) if hasattr(self.__model, 'predict_proba') else None

        # Coefficienti del modello
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = [f'PC{i + 1}' if self.__use_pca else f'feature_{i}'
                             for i in range(X_train.shape[1])]

        if len(self.__model.coef_.shape) == 2:
            if self.__model.coef_.shape[0] == 1:
                coefficients_values = self.__model.coef_[0]
            else:
                coefficients_values = self.__model.coef_
                feature_names = [f'{fn}_class_{i}' for i in range(self.__model.coef_.shape[0])
                                 for fn in feature_names]
        else:
            coefficients_values = self.__model.coef_

        if len(self.__model.coef_.shape) == 2 and self.__model.coef_.shape[0] == 1:
            coefficients = pd.DataFrame({
                'variabile': feature_names,
                'coefficiente': coefficients_values,
                'odds_ratio': np.exp(coefficients_values)
            }).sort_values('coefficiente', ascending=False)
        else:
            coefficients = None
            print("Nota: Modello multiclasse, coefficienti non visualizzati in DataFrame")

        intercept = self.__model.intercept_[0] if len(self.__model.intercept_) == 1 else self.__model.intercept_

        # Risultati completi
        self.__results = {
            'modello': self.__model,
            'coefficienti': coefficients,
            'intercept': intercept,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_proba_test': y_proba_test,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'use_pca': self.__use_pca,
            'params': {
                'C': C,
                'penalty': penalty,
                'solver': solver,
                'max_iter': max_iter,
                'class_weight': class_weight
            }
        }

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

        # Preprocessa i dati
        X_processed = self._preprocess_predict_data(X_new)

        return self.__model.predict_proba(X_processed)

    def grafico_coefficienti(self):
        """
        Grafico dei coefficienti e odds ratio (solo per classificazione binaria).
        """
        if self.__results is None:
            raise ValueError("Train the model first")

        if self.__results['coefficienti'] is None:
            raise ValueError("Grafico coefficienti disponibile solo per classificazione binaria")

        coeffs = self.__results['coefficienti'].head(15)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].barh(coeffs['variabile'], coeffs['coefficiente'], color='steelblue')
        axes[0].set_xlabel('Coefficiente')
        axes[0].set_title(f'Coefficienti del modello{" con PCA" if self.__use_pca else ""}')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)

        axes[1].barh(coeffs['variabile'], coeffs['odds_ratio'], color='coral')
        axes[1].set_xlabel('Odds Ratio')
        axes[1].set_title(f'Odds Ratio{" con PCA" if self.__use_pca else ""}')
        axes[1].axvline(x=1, color='red', linestyle='--', linewidth=1)

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

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
