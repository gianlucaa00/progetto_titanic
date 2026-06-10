import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from it.akron_valtellina.models.dataset_split import Split


class ScalerPCA:
    def __init__(self, X, y, n_components=None, dummies=False):
        self.__X = X
        self.__y = y
        self.__n_components = n_components
        self.__scaler = StandardScaler()
        self.__dummies = dummies
        self.__pca = None

    def get_scaling(self, dummies=None):
        # Usa il valore passato o quello di default dell'istanza
        use_dummies = dummies if dummies is not None else self.__dummies

        splitter = Split(self.__X, self.__y, dummies=use_dummies)
        X_train, X_test, y_train, y_test = splitter.get_split(dummies=use_dummies)

        X_train_scaled = self.__scaler.fit_transform(X_train)
        X_test_scaled = self.__scaler.transform(X_test)

        # Salva gli indici originali se esistono
        self.__X_train_index = X_train.index if hasattr(X_train, 'index') else None
        self.__X_test_index = X_test.index if hasattr(X_test, 'index') else None

        return X_train_scaled, X_test_scaled, y_train, y_test

    def get_pca(self, dummies=None, variance_threshold=0.95):
        """
        Applica PCA ai dati scalati.

        Parameters:
        -----------
        dummies : bool, optional
            Se applicare one-hot encoding
        variance_threshold : float, default=0.95
            Soglia di varianza spiegata (usata solo se n_components=None)
        """
        use_dummies = dummies if dummies is not None else self.__dummies

        X_train_scaled, X_test_scaled, y_train, y_test = self.get_scaling(dummies=use_dummies)

        if self.__n_components is None:
            # Seleziona le componenti che spiegano la soglia di varianza
            pca_temp = PCA()
            pca_temp.fit(X_train_scaled)

            # Calcola la varianza cumulata
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)

            # Trova il numero di componenti che raggiunge la soglia
            n_components_selected = np.argmax(cumsum_var >= variance_threshold) + 1

            self.__pca = PCA(n_components=n_components_selected)
        else:
            self.__pca = PCA(n_components=self.__n_components)

        # Applica PCA
        X_train_pca = self.__pca.fit_transform(X_train_scaled)
        X_test_pca = self.__pca.transform(X_test_scaled)

        # Crea nomi dinamici per le componenti
        n_components_actual = X_train_pca.shape[1]
        component_names = [f"PC{i + 1}" for i in range(n_components_actual)]

        # Gestisci gli indici per i DataFrame
        X_train_pca = pd.DataFrame(
            X_train_pca,
            columns=component_names,
            index=self.__X_train_index if self.__X_train_index is not None else range(len(X_train_pca))
        )

        X_test_pca = pd.DataFrame(
            X_test_pca,
            columns=component_names,
            index=self.__X_test_index if self.__X_test_index is not None else range(len(X_test_pca))
        )

        # Info PCA
        self.__pca_info = {
            'explained_variance_ratio_': self.__pca.explained_variance_ratio_,
            'cumulative_variance_': np.cumsum(self.__pca.explained_variance_ratio_),
            'n_components_': n_components_actual
        }

        return X_train_pca, X_test_pca, y_train, y_test

    def get_pca_info(self):
        """Restituisce informazioni sulla PCA eseguita"""
        if self.__pca_info is None:
            raise ValueError("Devi prima eseguire get_pca()")
        return self.__pca_info

    def transform_new_data(self, X_new):
        """
        Trasforma nuovi dati usando lo scaler e PCA già addestrati.
        """
        if self.__scaler is None or self.__pca is None:
            raise ValueError("Devi prima eseguire get_pca() per addestrare scaler e PCA")

        X_scaled = self.__scaler.transform(X_new)
        X_pca = self.__pca.transform(X_scaled)

        return X_pca