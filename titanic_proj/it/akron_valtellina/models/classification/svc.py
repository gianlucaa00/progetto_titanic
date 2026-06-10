from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from it.akron_valtellina.models.scaling_pca import ScalerPCA
from it.akron_valtellina.models.dataset_split import Split
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
        self._train_columns = None
        self.__label_encoder = None

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
            splitter = Split(self.__X, self.__y, dummies=True)
            X_train_raw, X_test_raw, y_train, y_test = splitter.get_split(dummies=True)
            self._train_columns = X_train_raw.columns.tolist() if hasattr(X_train_raw, 'columns') else None
            X_train = self.__scaler.fit_transform(X_train_raw)
            X_test = self.__scaler.transform(X_test_raw)

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

            if not self.__use_pca:
                return self.__scaler.transform(X_processed)

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
            raise ValueError("Questo modello SVM non supporta predict_proba (usa probability=True)")

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
