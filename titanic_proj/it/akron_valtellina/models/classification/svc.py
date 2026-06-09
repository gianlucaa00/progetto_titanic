from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from it.akron_valtellina.models.scaling_pca import ScalerPCA
import pandas as pd



class SupportVectorMachine:
    def __init__(self, X, y, use_pca=False, n_components=None):
        """
        Support Vector Machine Classifier.

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
        self.__scaler = StandardScaler()
        self.__use_pca = use_pca
        self.__n_components = n_components
        self.__pca_transformer = None  # Per salvare il trasformatore PCA

    def train(self, kernel='rbf', C=1.0, gamma='scale', class_weight=None, **kwargs):
        """
        Addestra il modello SVM.

        Parameters:
        -----------
        kernel : str, default='rbf'
            Tipo di kernel ('linear', 'poly', 'rbf', 'sigmoid')
        C : float, default=1.0
            Parametro di regolarizzazione
        gamma : str or float, default='scale'
            Coefficiente per kernel rbf/poly/sigmoid
        class_weight : dict or 'balanced', default=None
        **kwargs : parametri aggiuntivi per SVC
        """

        if self.__use_pca:
            # Usa PCA per riduzione dimensionalità
            pca_transformer = ScalerPCA(self.__X, self.__y, n_components=self.__n_components, dummies=True)
            X_train, X_test, y_train, y_test = pca_transformer.get_pca()
            self.__pca_transformer = pca_transformer  # Salva per future trasformazioni

            # Stampa info sulla varianza spiegata se disponibile
            try:
                pca_info = pca_transformer.get_pca_info()
                print(f"PCA: {pca_info['n_components_']} componenti selezionate")
                print(f"Varianza spiegata cumulata: {pca_info['cumulative_variance_'][-1]:.2%}")
            except:
                pass
        else:
            # Usa solo scaling senza PCA
            scaler = ScalerPCA(self.__X, self.__y, n_components=None, dummies=True)
            X_train, X_test, y_train, y_test = scaler.get_scaling(dummies=True)

        # Codifica il target se necessario
        if isinstance(y_train, pd.Series) and y_train.dtype == 'object':
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            self.__label_encoder = le  # Salva per future predizioni

        # Crea e addestra il modello SVC
        self.__model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=True,  # Utile per probabilità delle classi
            random_state=42,
            **kwargs
        )

        self.__model.fit(X_train, y_train)

        # Predizioni su train e test
        y_pred_train = self.__model.predict(X_train)
        y_pred_test = self.__model.predict(X_test)
        y_proba_test = None

        # Probabilità (solo se disponibile)
        if hasattr(self.__model, 'predict_proba'):
            y_proba_test = self.__model.predict_proba(X_test)

        # Salva risultati completi
        self.__results = {
            'modello': self.__model,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_proba_test': y_proba_test,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'use_pca': self.__use_pca,
            'params': {
                'kernel': kernel,
                'C': C,
                'gamma': gamma,
                'class_weight': class_weight
            }
        }

        # Aggiungi info PCA se disponibile
        if self.__use_pca and hasattr(self, '_ScalerPCA__pca_info'):
            self.__results['pca_info'] = pca_transformer.get_pca_info()

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
            # Se non è stata usata PCA, applica solo scaling
            from it.akron_valtellina.models.scaling_pca import ScalerPCA
            temp_scaler = ScalerPCA(X_new, None, dummies=True)
            X_transformed, _, _, _ = temp_scaler.get_scaling(dummies=True)
            # Nota: questo non è ottimale - idealmente salveresti lo scaler

        return self.__model.predict(X_transformed)

    def predict_proba(self, X_new):
        """
        Predice probabilità su nuovi dati.
        """
        if self.__model is None:
            raise ValueError("Devi prima addestrare il modello con train()")

        if not hasattr(self.__model, 'predict_proba'):
            raise ValueError("Questo modello SVM non supporta predict_proba (usa probability=True)")

        # Applica le stesse trasformazioni
        if self.__use_pca and self.__pca_transformer is not None:
            X_transformed = self.__pca_transformer.transform_new_data(X_new)
        else:
            from it.akron_valtellina.models.scaling_pca import ScalerPCA
            temp_scaler = ScalerPCA(X_new, None, dummies=True)
            X_transformed, _, _, _ = temp_scaler.get_scaling(dummies=True)

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