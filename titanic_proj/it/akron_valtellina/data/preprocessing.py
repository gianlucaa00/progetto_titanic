from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from it.akron_valtellina.models.dataset_split import Split
from scipy.stats import jarque_bera
from sklearn.impute import KNNImputer


class CleanDataset:
    def __init__(self, X, y):
        """
        Classe per il preprocessing dei dati.

        """

        self.__X = X
        self.__y = y
        self.__encoders = {}


    def replace_strange_missing_values(self, X=None):
        """Sostituisce valori mancanti strani con NaN."""
        df = X if X is not None else self.__X
        strange_tokens = ["?", "Unknown", "unknown", "NAN", "nan", "NA", "na", "", " ", "NULL", "null", "None", "none"]
        return df.replace(strange_tokens, np.nan)

    def individua_na(self):
        """Identifica i valori mancanti nel dataset."""
        df = self.replace_strange_missing_values()
        null_values = df.isna().sum()
        miss_percent = df.isna().sum() * 100 / len(df)

        return null_values, miss_percent

    def imputa_con_knn(self, X_train, X_test, n_neighbors=5, weights='uniform'):
        """
        Imputa i valori mancanti utilizzando KNN.

        Returns:
        --------
        tuple: (X_train_imputed, X_test_imputed, stat_dict)
        """

        # Step 1: Identifica colonne
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns

        # Step 2: Converti colonne categoriche
        encoders = {}
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()

        for col in categorical_cols:
            le = LabelEncoder()
            train_values = X_train[col].dropna().astype(str)

            if len(train_values.unique()) > 0:
                le.fit(train_values)
                X_train_encoded[col] = le.transform(X_train[col].astype(str))

                mode_value = train_values.mode()
                default_category = mode_value[0] if len(mode_value) > 0 else train_values.iloc[0] if len(
                    train_values) > 0 else 'unknown'

                def safe_transform(x):
                    try:
                        return le.transform([str(x)])[0]
                    except ValueError:
                        return le.transform([default_category])[0]

                X_test_encoded[col] = X_test[col].astype(str).apply(safe_transform)
                encoders[col] = le
            else:
                X_train_encoded[col] = 0
                X_test_encoded[col] = 0

        # Step 3: KNN Imputer
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        imputer.fit(X_train_encoded)

        X_train_imputed_array = imputer.transform(X_train_encoded)
        X_test_imputed_array = imputer.transform(X_test_encoded)

        # Step 4: Ricostruisci DataFrame
        X_train_imputed = pd.DataFrame(X_train_imputed_array, columns=X_train_encoded.columns, index=X_train.index)
        X_test_imputed = pd.DataFrame(X_test_imputed_array, columns=X_test_encoded.columns, index=X_test.index)

        # Step 5: Reconverti colonne categoriche
        for col in categorical_cols:
            if col in encoders:
                train_rounded = np.clip(np.round(X_train_imputed[col]).astype(int), 0, len(encoders[col].classes_) - 1)
                test_rounded = np.clip(np.round(X_test_imputed[col]).astype(int), 0, len(encoders[col].classes_) - 1)
                X_train_imputed[col] = encoders[col].inverse_transform(train_rounded)
                X_test_imputed[col] = encoders[col].inverse_transform(test_rounded)

        # Step 6: Statistiche
        stat = {
            "na_before_train": int(X_train.isna().sum().sum()),
            "na_before_test": int(X_test.isna().sum().sum()),
            "na_after_train": int(X_train_imputed.isna().sum().sum()),
            "na_after_test": int(X_test_imputed.isna().sum().sum()),
            "n_neighbors": n_neighbors,
            "weights": weights,
            "numeric_cols": len(numeric_cols),
            "categorical_cols": len(categorical_cols)
        }

        return X_train_imputed, X_test_imputed, stat

    def gestione_na(self, test_size=0.2, random_state=42, soglia_na=50.0,
                    imputation_method='statistical', knn_neighbors=5):
        """
        Elimina e imputa i valori mancanti.

        Returns:
        --------
        dict: {
            'X': array-like,
            'y': array-like,
            'preprocessing_info': dict
        }
        """
        df = self.replace_strange_missing_values()

        # 1. elimina colonne con troppi NA
        na_percent = df.isna().sum() * 100 / len(df)
        cols_da_tenere = df.columns[na_percent <= soglia_na]
        cols_eliminate = df.columns[na_percent > soglia_na]
        df = df[cols_da_tenere]

        # 2. elimina duplicati
        df = df.drop_duplicates()

        # 3. split PRIMA dell'imputazione
        if self.__y is not None:
            y_aligned = self.__y.loc[df.index] if hasattr(self.__y, 'loc') else self.__y[df.index]

            # Usa la classe Split
            splitter = Split(df, y_aligned, dummies=False, test_size=test_size, random_state=random_state)
            X_train, X_test, y_train, y_test = splitter.get_split(dummies=False)
        else:
            splitter = Split(df, None, dummies=False, test_size=test_size, random_state=random_state)
            X_train, X_test, _, _ = splitter.get_split(dummies=False)
            y_train, y_test = None, None

        # 4. Imputazione
        knn_stat = None
        if imputation_method == 'knn':
            X_train, X_test, knn_stat = self.imputa_con_knn(X_train, X_test, n_neighbors=knn_neighbors)
        else:
            fill_values = {}
            for col in X_train.columns:
                if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'string' or X_train[
                    col].dtype.name == 'category':
                    mode_val = X_train[col].mode()
                    fill_values[col] = mode_val[0] if len(mode_val) > 0 else (
                        X_train[col].iloc[0] if len(X_train[col]) > 0 else 0)
                elif pd.api.types.is_numeric_dtype(X_train[col]):
                    col_clean = X_train[col].dropna()
                    if len(col_clean) > 1:
                        try:
                            p_value = jarque_bera(col_clean)[1]
                            fill_values[col] = X_train[col].mean() if p_value > 0.05 else X_train[col].median()
                        except:
                            fill_values[col] = X_train[col].median()
                    elif len(col_clean) == 1:
                        fill_values[col] = col_clean.iloc[0]
                    else:
                        fill_values[col] = 0
            X_train = X_train.fillna(fill_values)
            X_test = X_test.fillna(fill_values)

        # 5. elimina colonne costanti
        const_cols = X_train.columns[X_train.nunique(dropna=False) > 1]
        X_train = X_train[const_cols]
        X_test = X_test[const_cols]

        # 6. ricompone dataset
        if y_train is not None and y_test is not None:
            y_train = y_train.loc[X_train.index]
            y_test = y_test.loc[X_test.index]
            X = pd.concat([X_train, X_test], axis=0).sort_index()
            y = pd.concat([y_train, y_test], axis=0).sort_index()
            self.__X = X
            self.__y = y
        else:
            X = pd.concat([X_train, X_test], axis=0).sort_index()
            self.__X = X
            self.__y = None

        # Risultato in formato JSON
        result = {
            'X': self.__X.to_dict(orient='records') if self.__X is not None else None,
            'y': self.__y.tolist() if self.__y is not None else None,
            'preprocessing_info': {
                'imputation_method': imputation_method,
                'columns_removed_high_na': len(cols_eliminate),
                'duplicates_removed': int(df.duplicated().sum()),
                'final_shape': list(self.__X.shape) if self.__X is not None else None,
                'constant_columns_removed': len(const_cols) if len(const_cols) > 0 else 0
            }
        }

        if knn_stat:
            result['preprocessing_info']['knn_stats'] = knn_stat

        return self.__X, self.__y, result

    def trova_outliers(self, method='iqr', threshold=1.5, imputation_method='statistical', knn_neighbors=5):
        """
        Trova e rimuove gli outliers.

        Parameters:
        -----------
        method : str, default='iqr'
            Metodo per outliers: 'iqr' o 'zscore'
        threshold : float, default=1.5
            Soglia per IQR (1.5) o numero di deviazioni standard per zscore (3)
        """
        df, y, _ = self.gestione_na(imputation_method=imputation_method, knn_neighbors=knn_neighbors)

        if df is None:
            raise ValueError("Errore nel preprocessing dei dati")

        initial_shape = df.shape

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        outliers_removed = 0

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_removed += outliers_mask.sum()
                df = df[~outliers_mask]

            elif method == 'zscore':
                from scipy.stats import zscore
                z_scores = np.abs(zscore(df[col].dropna()))
                outliers_mask = z_scores > threshold
                outliers_removed += outliers_mask.sum()
                df = df[~outliers_mask]

        if y is not None:
            y_final = y.loc[df.index]
        else:
            y_final = None

        results = {
            "initial_shape": initial_shape,
            "outliers_rimossi": outliers_removed,
            "final_shape": df.shape
        }

        return df, y_final, results

    def elimina_colonne(self, columns_to_drop=None):
        """
        Elimina colonne specificate dal dataset.

        Parameters:
        -----------
        columns_to_drop : list, optional
            Lista di nomi di colonne da eliminare

        Returns:
        --------
        df nuovo
        """
        if self.__X is None:
            raise ValueError("Nessun dato caricato. Esegui prima il caricamento dei dati.")

        columns_dropped = []
        columns_kept = list(self.__X.columns)

        # Elimina colonne specifiche
        if columns_to_drop is not None:
            if isinstance(columns_to_drop, str):
                columns_to_drop = [columns_to_drop]

            for col in columns_to_drop:
                if col in self.__X.columns:
                    columns_dropped.append(col)
                    columns_kept.remove(col)

            self.__X = self.__X.drop(columns=[col for col in columns_to_drop if col in self.__X.columns])

        return self.__X, self.__y, columns_dropped




