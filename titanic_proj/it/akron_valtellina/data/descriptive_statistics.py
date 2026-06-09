import pandas as pd
from it.akron_valtellina.data.preprocessing import CleanDataset


class StatisticaDescrittiva:
    """
    Classe per l'analisi statistica descrittiva dei dati.
    Le statistiche sono calcolate SOLO sul dataset originale (non preprocessato).
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
        Calcola le statistiche descrittive del dataset ORIGINALE.
        Nessun preprocessing applicato.

        Returns:
        --------
        dict: Statistiche descrittive in formato JSON
        """
        # Ottieni i dati originali (solo replace dei valori strani con NaN)
        df_original = self.__clean_dataset.replace_strange_missing_values()

        # Ottieni informazioni sui valori mancanti sui dati originali
        null_values, miss_percent = self.__clean_dataset.individua_na()

        # Informazioni sul target originale
        y_original = self.__clean_dataset._CleanDataset__y

        # Calcola statistiche per colonne numeriche (solo su dati originali)
        numeric_stats = {}
        numeric_cols = df_original.select_dtypes(include=['int64', 'float64']).columns

        for col in numeric_cols:
            # Rimuovi NaN per i calcoli statistici
            clean_col = df_original[col].dropna()

            if len(clean_col) > 0:
                numeric_stats[col] = {
                    'mean': float(clean_col.mean()) if not pd.isna(clean_col.mean()) else None,
                    'median': float(clean_col.median()) if not pd.isna(clean_col.median()) else None,
                    'std': float(clean_col.std()) if not pd.isna(clean_col.std()) else None,
                    'min': float(clean_col.min()) if not pd.isna(clean_col.min()) else None,
                    'max': float(clean_col.max()) if not pd.isna(clean_col.max()) else None,
                    'q1': float(clean_col.quantile(0.25)) if not pd.isna(clean_col.quantile(0.25)) else None,
                    'q3': float(clean_col.quantile(0.75)) if not pd.isna(clean_col.quantile(0.75)) else None,
                    'skewness': float(clean_col.skew()) if not pd.isna(clean_col.skew()) else None,
                    'kurtosis': float(clean_col.kurtosis()) if not pd.isna(clean_col.kurtosis()) else None,
                    'n_missing': int(df_original[col].isna().sum()),
                    'missing_percent': float(df_original[col].isna().sum() * 100 / len(df_original))
                }
            else:
                numeric_stats[col] = {
                    'mean': None,
                    'median': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'q1': None,
                    'q3': None,
                    'skewness': None,
                    'kurtosis': None,
                    'n_missing': int(df_original[col].isna().sum()),
                    'missing_percent': float(df_original[col].isna().sum() * 100 / len(df_original)),
                    'warning': 'Colonna vuota (solo valori mancanti)'
                }

        # Calcola statistiche per colonne categoriche (solo su dati originali)
        categorical_stats = {}
        categorical_cols = df_original.select_dtypes(include=['object', 'string', 'category']).columns

        for col in categorical_cols:
            # Rimuovi NaN per il conteggio
            clean_col = df_original[col].dropna()
            value_counts = clean_col.value_counts()

            categorical_stats[col] = {
                'unique_values': int(clean_col.nunique()) if len(clean_col) > 0 else 0,
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'most_frequent_percent': float(value_counts.iloc[0] * 100 / len(clean_col)) if len(
                    value_counts) > 0 and len(clean_col) > 0 else 0,
                'frequencies': {str(k): int(v) for k, v in value_counts.head(10).to_dict().items()},
                'n_missing': int(df_original[col].isna().sum()),
                'missing_percent': float(df_original[col].isna().sum() * 100 / len(df_original))
            }

        # Informazioni sul target originale
        target_info = {}
        if y_original is not None:
            # Gestisci il caso in cui y sia una Serie o DataFrame
            if isinstance(y_original, pd.DataFrame):
                y_series = y_original.iloc[:, 0] if y_original.shape[1] == 1 else y_original
            else:
                y_series = y_original

            # Rimuovi NaN
            y_clean = y_series.dropna()

            if pd.api.types.is_numeric_dtype(y_clean):
                target_info = {
                    'type': 'numeric',
                    'mean': float(y_clean.mean()) if not pd.isna(y_clean.mean()) else None,
                    'std': float(y_clean.std()) if not pd.isna(y_clean.std()) else None,
                    'min': float(y_clean.min()) if not pd.isna(y_clean.min()) else None,
                    'max': float(y_clean.max()) if not pd.isna(y_clean.max()) else None,
                    'n_missing': int(y_series.isna().sum()),
                    'missing_percent': float(y_series.isna().sum() * 100 / len(y_series))
                }
            else:
                value_counts = y_clean.value_counts()
                target_info = {
                    'type': 'categorical',
                    'unique_values': int(y_clean.nunique()),
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'class_distribution': {str(k): int(v) for k, v in value_counts.to_dict().items()},
                    'class_balance': {str(k): round(float(v / len(y_clean)), 4) for k, v in
                                      value_counts.to_dict().items()},
                    'n_missing': int(y_series.isna().sum()),
                    'missing_percent': float(y_series.isna().sum() * 100 / len(y_series))
                }

        # Risultato finale
        result_stats = {
            'dataset_info': {
                'name': 'Original Dataset (no preprocessing)',
                'shape': list(df_original.shape),
                'columns': df_original.columns.tolist(),
                'dtypes': {k: str(v) for k, v in df_original.dtypes.to_dict().items()},
                'total_cells': int(df_original.shape[0] * df_original.shape[1])
            },
            'missing_values': {
                'null_values': null_values.to_dict() if hasattr(null_values, 'to_dict') else null_values,
                'missing_percent': miss_percent.to_dict() if hasattr(miss_percent, 'to_dict') else miss_percent,
                'total_missing': int(null_values.sum()) if hasattr(null_values, 'sum') else 0,
                'total_missing_percent': float(
                    null_values.sum() * 100 / (df_original.shape[0] * df_original.shape[1])) if hasattr(null_values,
                                                                                                        'sum') else 0,
                'columns_with_missing': int((null_values > 0).sum()) if hasattr(null_values, 'gt') else 0
            },
            'numeric_columns_stats': numeric_stats,
            'categorical_columns_stats': categorical_stats,
            'target_info': target_info,
            'summary': {
                'n_rows': int(df_original.shape[0]),
                'n_columns': int(df_original.shape[1]),
                'n_numeric_columns': len(numeric_cols),
                'n_categorical_columns': len(categorical_cols),
                'n_missing_values_total': int(df_original.isna().sum().sum()),
                'missing_percent_total': float(
                    df_original.isna().sum().sum() * 100 / (df_original.shape[0] * df_original.shape[1]))
            }
        }

        return result_stats

    def quick_summary(self):
        """
        Restituisce un riepilogo rapido del dataset ORIGINALE.

        Returns:
        --------
        dict: Riepilogo rapido
        """
        df = self.__clean_dataset.replace_strange_missing_values()

        if df is None:
            return {'error': 'Nessun dato caricato'}

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns

        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()

        return {
            'dataset': 'original (no preprocessing)',
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'missing_values_total': int(missing_cells),
            'missing_percent': round(float(missing_cells * 100 / total_cells), 2) if total_cells > 0 else 0,
            'duplicates': int(df.duplicated().sum()),
            'memory_usage_mb': round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 2)
        }

    def get_column_info(self, column_name):
        """
        Restituisce statistiche dettagliate per una colonna specifica.

        Parameters:
        -----------
        column_name : str
            Nome della colonna da analizzare

        Returns:
        --------
        dict: Statistiche della colonna
        """
        df = self.__clean_dataset.replace_strange_missing_values()

        if column_name not in df.columns:
            return {'error': f'Colonna "{column_name}" non trovata'}

        col_data = df[column_name].dropna()

        result = {
            'column_name': column_name,
            'dtype': str(df[column_name].dtype),
            'n_total': int(len(df[column_name])),
            'n_missing': int(df[column_name].isna().sum()),
            'missing_percent': float(df[column_name].isna().sum() * 100 / len(df[column_name])),
            'n_unique': int(col_data.nunique()) if len(col_data) > 0 else 0
        }

        # Statistiche per colonne numeriche
        if pd.api.types.is_numeric_dtype(col_data) and len(col_data) > 0:
            result['numeric_stats'] = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'q1': float(col_data.quantile(0.25)),
                'q3': float(col_data.quantile(0.75)),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis())
            }

        # Statistiche per colonne categoriche
        elif len(col_data) > 0:
            value_counts = col_data.value_counts()
            result['categorical_stats'] = {
                'most_frequent': str(value_counts.index[0]),
                'most_frequent_count': int(value_counts.iloc[0]),
                'most_frequent_percent': round(float(value_counts.iloc[0] * 100 / len(col_data)), 2),
                'top_5_values': {str(k): int(v) for k, v in value_counts.head(5).to_dict().items()}
            }

        return result