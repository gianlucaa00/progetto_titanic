from ucimlrepo import fetch_ucirepo
import pandas as pd
import os


class Dataset:
    def __init__(self, source='uci', dataset_id=73, csv_path=None, target_column=None):
        """
        Classe per il preprocessing dei dati.

        Parameters:
        -----------
        source : str, default='uci'
            Fonte dei dati: 'uci', 'csv'
        dataset_id : int, default=73
            ID del dataset UCI (es. 73 per Iris, 186 per Wine)
        csv_path : str, optional
            Percorso del file CSV (se source='csv')
        target_column : str, optional
            Nome della colonna target (se source='csv')
        """
        self.__source = source
        self.__dataset_id = dataset_id
        self.__csv_path = csv_path
        self.__target_column = target_column
        self.__X = None
        self.__y = None
        self.__encoders = {}

        # Carica i dati in base alla fonte
        self.__load_data()

    def __load_data(self):
        """Carica i dati dalla fonte specificata."""
        if self.__source == 'csv':
            self.__load_from_csv()
        else:  # default uci
            self.__load_from_uci()

    def __load_from_uci(self):
        """Carica i dati da UCI ML Repository."""
        dataset = fetch_ucirepo(id=self.__dataset_id)
        self.__X = dataset.data.features
        self.__y = dataset.data.targets

        # Gestisci il caso in cui y sia una Series o DataFrame
        if self.__y is not None and isinstance(self.__y, pd.DataFrame):
            if self.__y.shape[1] == 1:
                self.__y = self.__y.iloc[:, 0]


    def __load_from_csv(self):
        """Carica i dati da file CSV."""
        if self.__csv_path is None:
            raise ValueError("Per caricare da CSV, specificare 'csv_path'")

        if not os.path.exists(self.__csv_path):
            raise FileNotFoundError(f"File non trovato: {self.__csv_path}")

        df = pd.read_csv(self.__csv_path)

        # Gestione target column
        if self.__target_column:
            if self.__target_column not in df.columns:
                raise ValueError(
                    f"Colonna target '{self.__target_column}' non trovata. Colonne disponibili: {list(df.columns)}")

            self.__y = df[self.__target_column]
            self.__X = df.drop(columns=[self.__target_column])

        else:
            # Se non specificato, assume che l'ultima colonna sia il target
            self.__y = df.iloc[:, -1]
            self.__X = df.iloc[:, :-1]

    def get_data(self):
        """Restituisce X e y originali."""
        return self.__X, self.__y

    def get_info(self):
        """Restituisce informazioni sul dataset."""
        if self.__X is None:
            return "Nessun dato caricato"

        info = {
            'source': self.__source,
            'shape': self.__X.shape,
            'columns': list(self.__X.columns),
            'numeric_cols': list(self.__X.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_cols': list(self.__X.select_dtypes(include=['object', 'string', 'category']).columns),
            'has_target': self.__y is not None
        }

        if self.__y is not None:
            info['target_shape'] = self.__y.shape
            if isinstance(self.__y, pd.Series):
                info['target_unique'] = self.__y.nunique()

        return info

    def create_cleaner(self):
        """
        Crea un'istanza di CleanDataset con i dati caricati.

        Returns:
        --------
        CleanDataset: Istanza per il preprocessing
        """
        return CleanDataset(self.__X, self.__y)