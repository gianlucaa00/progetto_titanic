from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from it.akron_valtellina.models.dataset_split import Split
from it.akron_valtellina.models.scaling_pca import ScalerPCA
import pandas as pd


class Regressione:
    def __init__(self, X, y, metodo="elasticnet", use_pca=False, n_components=None):
        """
        Regressione Lineare con diversi metodi (lineare, ridge, lasso, elasticnet).

        Parameters:
        -----------
        X : DataFrame or array-like
            Feature matrix
        y : Series or array-like
            Target variable
        metodo : str, default='elasticnet'
            Tipo di regressione: 'lineare', 'ridge', 'lasso', 'elasticnet'
        use_pca : bool, default=False
            Se True, applica PCA prima dell'addestramento
        n_components : int or None, default=None
            Numero di componenti PCA (se None e use_pca=True, seleziona automaticamente 95% varianza)
        """
        self.__X = X
        self.__y = y
        self.__model = None
        self.__results = None
        self.__metodo = metodo
        self.__use_pca = use_pca
        self.__n_components = n_components
        self.__pca_transformer = None
        self.__scaler = None

    def train(self, alpha=1.0, l1_ratio=0.5, **kwargs):
        """
        Addestra il modello di regressione.

        Parameters:
        -----------
        alpha : float, default=1.0
            Parametro di regolarizzazione (per ridge, lasso, elasticnet)
        l1_ratio : float, default=0.5
            Rapporto L1 per elasticnet (0 = ridge, 1 = lasso)
        **kwargs : parametri aggiuntivi per il modello specifico
        """

        # Gestione PCA vs scaling semplice
        if self.__use_pca:
            # Applica PCA
            pca_transformer = ScalerPCA(self.__X, self.__y, n_components=self.__n_components, dummies=False)
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
            # Solo split con scaling
            splitter = Split(self.__X, self.__y, dummies=False)
            X_train, X_test, y_train, y_test = splitter.get_split(dummies=False)

            # Scaling delle feature
            self.__scaler = StandardScaler()
            X_train = self.__scaler.fit_transform(X_train)
            X_test = self.__scaler.transform(X_test)

        # Selezione del modello in base al metodo
        if self.__metodo == "lineare":
            self.__model = LinearRegression(**kwargs)
        elif self.__metodo == "ridge":
            self.__model = Ridge(alpha=alpha, **kwargs)
        elif self.__metodo == "lasso":
            self.__model = Lasso(alpha=alpha, **kwargs)
        else:  # elasticnet
            self.__model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)

        # Addestra il modello
        self.__model.fit(X_train, y_train)

        # Predizioni
        y_pred_train = self.__model.predict(X_train)
        y_pred_test = self.__model.predict(X_test)


        # Coefficienti del modello
        coefficients = None
        if hasattr(self.__model, 'coef_'):
            if hasattr(X_train, 'columns') and not self.__use_pca:
                feature_names = X_train.columns
            elif self.__use_pca and hasattr(X_train, 'columns'):
                feature_names = X_train.columns
            else:
                feature_names = [f'PC{i + 1}' if self.__use_pca else f'feature_{i}'
                                 for i in range(len(self.__model.coef_))]

            coefficients = pd.DataFrame({
                'variabile': feature_names,
                'coefficiente': self.__model.coef_
            }).sort_values('coefficiente', ascending=False)

        # Risultati completi
        self.__results = {
            'modello': self.__model,
            'coefficienti': coefficients,
            'intercetta': self.__model.intercept_ if hasattr(self.__model, 'intercept_') else None,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'use_pca': self.__use_pca,
            'metodo': self.__metodo,
            'params': {
                'alpha': alpha if self.__metodo != 'lineare' else None,
                'l1_ratio': l1_ratio if self.__metodo == 'elasticnet' else None
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


