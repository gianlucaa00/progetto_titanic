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

    def train(self, C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, class_weight=None, **kwargs):
        """
        Addestra il modello di regressione logistica.

        Parameters:
        -----------
        C : float, default=1.0
            Inverse of regularization strength
        penalty : str, default='l2'
            Norm used in the penalization ('l1', 'l2', 'elasticnet', None)
        solver : str, default='lbfgs'
            Algorithm to use in optimization problem
        max_iter : int, default=1000
            Maximum number of iterations
        class_weight : dict or 'balanced', default=None
            Weights associated with classes
        **kwargs : parametri aggiuntivi per LogisticRegression
        """

        # Gestione PCA vs splitting semplice
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
            # Solo split
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

        # Coefficienti del modello (gestione multiclasse)
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = [f'PC{i + 1}' if self.__use_pca else f'feature_{i}'
                             for i in range(X_train.shape[1])]

        # Gestione coefficienti per classificazione binaria vs multiclasse
        if len(self.__model.coef_.shape) == 2:
            if self.__model.coef_.shape[0] == 2:
                # Per classificazione binaria, prendi i coefficienti della classe positiva
                coefficients_values = self.__model.coef_[1]
            else:
                # Per multiclasse con più di 2 classi
                coefficients_values = self.__model.coef_
                feature_names = [f'{fn}_class_{i}' for i in range(self.__model.coef_.shape[0])
                                 for fn in feature_names]
        else:
            coefficients_values = self.__model.coef_

        # Crea DataFrame dei coefficienti solo per classificazione binaria
        if len(self.__model.coef_.shape) == 2 and self.__model.coef_.shape[0] == 2:
            coefficients = pd.DataFrame({
                'variabile': feature_names,
                'coefficiente': coefficients_values,
                'odds_ratio': np.exp(coefficients_values)
            }).sort_values('coefficiente', ascending=False)
        else:
            # Per multiclasse, salva i coefficienti in modo diverso
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

        # Grafico coefficienti
        axes[0].barh(coeffs['variabile'], coeffs['coefficiente'], color='steelblue')
        axes[0].set_xlabel('Coefficiente')
        axes[0].set_title(f'Coefficienti del modello{" con PCA" if self.__use_pca else ""}')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)

        # Grafico odds ratio
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