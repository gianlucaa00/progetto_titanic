import pandas as pd
from it.akron_valtellina.data.preprocessing import CleanDataset


class StatisticaDescrittiva:
    """
    Classe per l'analisi statistica descrittiva dei dati.
    """

    def __init__(self, clean_dataset_instance):
        """
        Parameters:
        -----------
        clean_dataset_instance : CleanDataset
            Istanza della classe CleanDataset già caricata
        """
        if not isinstance(clean_dataset_instance, CleanDataset):
            raise ValueError("clean_dataset_instance deve essere un'istanza di CleanDataset")

        self.__clean_dataset = clean_dataset_instance

    def statistica_descrittiva(self):
        """
        Calcola le statistiche descrittive del dataset.

        Returns:
        --------
        dict: Statistiche descrittive in formato JSON
        """
        # Ottieni i dati originali con valori mancanti strani sostituiti
        df_old = self.__clean_dataset.replace_strange_missing_values()

        # Ottieni i dati dopo la gestione dei valori mancanti
        result = self.__clean_dataset.gestione_na()
        X = result['X_train'] if 'X_train' in result else result.get('X', None)
        y = result.get('y_train', result.get('y', None))

        if X is None:
            # Se gestione_na non ha funzionato, usa elimina_na
            X, y = self.__clean_dataset.elimina_na()

        df = X

        # Ottieni informazioni sui valori mancanti
        null_values, miss_percent = self.__clean_dataset.individua_na()

        # Calcola statistiche per colonne numeriche
        numeric_stats = {}
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        for col in numeric_cols:
            numeric_stats[col] = {
                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'q1': float(df[col].quantile(0.25)) if not pd.isna(df[col].quantile(0.25)) else None,
                'q3': float(df[col].quantile(0.75)) if not pd.isna(df[col].quantile(0.75)) else None,
                'skewness': float(df[col].skew()) if not pd.isna(df[col].skew()) else None,
                'kurtosis': float(df[col].kurtosis()) if not pd.isna(df[col].kurtosis()) else None
            }

        # Calcola statistiche per colonne categoriche
        categorical_stats = {}
        categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns

        for col in categorical_cols:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_values': int(df[col].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'frequencies': {str(k): int(v) for k, v in value_counts.head(10).to_dict().items()}
            }

        # Informazioni sul target
        target_info = {}
        if y is not None:
            if isinstance(y, pd.Series):
                if y.dtype in ['int64', 'float64']:
                    target_info = {
                        'type': 'numeric',
                        'mean': float(y.mean()) if not pd.isna(y.mean()) else None,
                        'std': float(y.std()) if not pd.isna(y.std()) else None,
                        'min': float(y.min()) if not pd.isna(y.min()) else None,
                        'max': float(y.max()) if not pd.isna(y.max()) else None
                    }
                else:
                    value_counts = y.value_counts()
                    target_info = {
                        'type': 'categorical',
                        'unique_values': int(y.nunique()),
                        'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                        'class_distribution': {str(k): int(v) for k, v in value_counts.to_dict().items()}
                    }

        # Risultato finale
        result_stats = {
            'dataset_info': {
                'original_shape': list(df_old.shape),
                'original_columns': df_old.columns.tolist(),
                'original_dtypes': {k: str(v) for k, v in df_old.dtypes.to_dict().items()},
                'processed_shape': list(df.shape),
                'processed_columns': df.columns.tolist(),
                'processed_dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()}
            },
            'missing_values': {
                'null_values': null_values.to_dict() if hasattr(null_values, 'to_dict') else null_values,
                'missing_percent': miss_percent.to_dict() if hasattr(miss_percent, 'to_dict') else miss_percent,
                'total_missing': int(null_values.sum()) if hasattr(null_values, 'sum') else 0,
                'columns_with_missing': int((null_values > 0).sum()) if hasattr(null_values, 'gt') else 0
            },
            'numeric_columns_stats': numeric_stats,
            'categorical_columns_stats': categorical_stats,
            'target_info': target_info,
            'summary': {
                'n_rows': int(df.shape[0]),
                'n_columns': int(df.shape[1]),
                'n_numeric_columns': len(numeric_cols),
                'n_categorical_columns': len(categorical_cols),
                'n_missing_values_total': int(df.isna().sum().sum())
            }
        }

        return result_stats

    def report_completo(self):
        """
        Genera un report statistico completo.

        Returns:
        --------
        dict: Report completo con tutte le statistiche
        """
        stats = self.statistica_descrittiva()

        # Aggiungi informazioni aggiuntive
        df = self.__clean_dataset.__X

        if df is not None:
            # Correlazioni tra variabili numeriche
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                stats['correlation_matrix'] = {
                    'columns': numeric_cols.tolist(),
                    'values': corr_matrix.values.tolist()
                }

            # Duplicati
            stats['duplicates'] = {
                'n_duplicates': int(df.duplicated().sum()),
                'duplicate_percent': float(df.duplicated().sum() * 100 / len(df)) if len(df) > 0 else 0
            }

        return stats

    def quick_summary(self):
        """
        Restituisce un riepilogo rapido del dataset.

        Returns:
        --------
        dict: Riepilogo rapido
        """
        df = self.__clean_dataset.__X

        if df is None:
            return {'error': 'Nessun dato caricato'}

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns

        return {
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'missing_values_total': int(df.isna().sum().sum()),
            'missing_percent': float(df.isna().sum().sum() * 100 / (df.shape[0] * df.shape[1])) if df.shape[0] *
                                                                                                   df.shape[
                                                                                                       1] > 0 else 0,
            'duplicates': int(df.duplicated().sum()),
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }


