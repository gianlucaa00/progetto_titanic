import numpy as np
import pandas as pd
import os
import pickle
from io import StringIO, BytesIO
from flask import Response
from flask import Flask, jsonify, request, send_file
from it.akron_valtellina.data.load_data import Dataset
from it.akron_valtellina.data.preprocessing import CleanDataset
from it.akron_valtellina.data.graphs import Visualizzatore
from it.akron_valtellina.data.descriptive_statistics import StatisticaDescrittiva
from it.akron_valtellina.models.classification.logistic_regression import RegressioneLogistica
from it.akron_valtellina.models.classification.svc import SupportVectorMachine
from it.akron_valtellina.models.classification.random_forest import RandomForest
from it.akron_valtellina.models.classification.xgboost import XGBoost
from it.akron_valtellina.models.kfold_crossvalidation import KFoldCrossValidation
from it.akron_valtellina.models.finetuning_iperparametri import FineTuning
from it.akron_valtellina.models.evaluation import Evaluation


app = Flask(__name__)

# Configurazione globale
# In Docker monta un volume e imposta, ad esempio:
#   TITANIC_DATA_DIR=/app/dataset
# oppure direttamente:
#   TITANIC_TRAIN_CSV=/app/dataset/train.csv
#   TITANIC_TEST_CSV=/app/dataset/test.csv
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.environ.get('TITANIC_DATA_DIR', os.path.join(PROJECT_ROOT, 'dataset'))
PROCESSED_X_PATH = os.environ.get('TITANIC_PROCESSED_X_CSV', os.path.join(DATA_DIR, 'processed_X.csv'))
PROCESSED_Y_PATH = os.environ.get('TITANIC_PROCESSED_Y_CSV', os.path.join(DATA_DIR, 'processed_y.csv'))
PROCESSED_INFO_PATH = os.environ.get('TITANIC_PREPROCESS_INFO_PKL', os.path.join(DATA_DIR, 'preprocess_info.pkl'))


def resolve_data_path(env_var, default_filename, legacy_paths=None):
    """Resolve dataset paths with env/volume first and local fallbacks for development."""
    env_path = os.environ.get(env_var)
    if env_path:
        return os.path.abspath(os.path.expanduser(env_path))

    candidates = [os.path.join(DATA_DIR, default_filename)]
    candidates.extend(legacy_paths or [])

    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(os.path.expanduser(path))

    return os.path.abspath(os.path.expanduser(candidates[0]))

DATASET_CONFIG = {
    'source': 'csv',  # 'uci' o 'csv'
    'csv_path': resolve_data_path(
        'TITANIC_TRAIN_CSV',
        'train.csv',
        [
            r"C:\Users\alisi\Downloads\train_processed.csv",
            r"C:\Users\alisi\Downloads\train.csv",
        ],
    ),
    'target_column': 'Survived'  # Modifica con il nome della colonna target
}
TEST_CONFIG = {
    'source': 'csv',  # 'uci' o 'csv'
    'csv_path': resolve_data_path(
        'TITANIC_TEST_CSV',
        'test.csv',
        [r"C:\Users\alisi\Downloads\test.csv"],
    )
}

# Variabili globali
_processed_X = None
_processed_y = None
_is_processed = False
_current_preprocess_params = {}
_current_preprocess_info = None
_models = {}
_selected_model = None
_test_df = None
_test_processed = None


def get_data():
    """Carica i dati e restituisce l'istanza di CleanDataset"""
    dataset = Dataset(
        source=DATASET_CONFIG['source'],
        csv_path=DATASET_CONFIG['csv_path'],
        target_column=DATASET_CONFIG['target_column'],
        is_test = False
    )
    X, y = dataset.get_data()
    cleaner = CleanDataset(X, y)
    return cleaner, dataset.get_info()

def get_test():
    test = Dataset(
        source=TEST_CONFIG['source'],
        csv_path=TEST_CONFIG['csv_path'],
        is_test = True
    )
    X_test = test.get_test()
    return X_test, test.get_info()

def get_processed_data_cached():
    """
    Restituisce i dati preprocessati dalla cache.
    """
    global _processed_X, _processed_y, _is_processed, _current_preprocess_info

    if _is_processed and _processed_X is not None:
        print("Usando dati preprocessati dalla cache")
        return _processed_X.copy(), _processed_y.copy()

    persisted = load_processed_data_from_volume()
    if persisted is not None:
        _processed_X, _processed_y = persisted
        _current_preprocess_info = load_preprocess_info_from_volume()
        _is_processed = True
        print("Usando dati preprocessati dal volume")
        return _processed_X.copy(), _processed_y.copy()

    print("Nessun dato in cache. Esegui prima /dataset/preprocess con save_state=true")
    return None, None


def save_processed_data_to_volume(X, y):
    """Persist processed data in DATA_DIR, useful when the app runs in Docker with a mounted volume."""
    if X is None or y is None:
        return

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        X.to_csv(PROCESSED_X_PATH, index=False)
        pd.Series(y, name='target').to_csv(PROCESSED_Y_PATH, index=False)
        print(f"Dataset preprocessato salvato su volume: {PROCESSED_X_PATH}, {PROCESSED_Y_PATH}")
    except Exception as exc:
        print(f"Impossibile salvare il dataset preprocessato su volume: {exc}")


def save_preprocess_info_to_volume(preprocess_info):
    """Persist preprocessing metadata, including KNN objects, for Docker volume based runs."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(PROCESSED_INFO_PATH, 'wb') as file:
            pickle.dump(preprocess_info, file)
        print(f"Info preprocessing salvate su volume: {PROCESSED_INFO_PATH}")
    except Exception as exc:
        print(f"Impossibile salvare info preprocessing su volume: {exc}")


def load_preprocess_info_from_volume():
    """Load preprocessing metadata from DATA_DIR if available."""
    try:
        if os.path.isfile(PROCESSED_INFO_PATH):
            with open(PROCESSED_INFO_PATH, 'rb') as file:
                return pickle.load(file)
    except Exception as exc:
        print(f"Impossibile caricare info preprocessing dal volume: {exc}")

    return None


def load_processed_data_from_volume():
    """Load processed data from DATA_DIR if available."""
    try:
        if os.path.isfile(PROCESSED_X_PATH) and os.path.isfile(PROCESSED_Y_PATH):
            X = pd.read_csv(PROCESSED_X_PATH)
            y = pd.read_csv(PROCESSED_Y_PATH).iloc[:, 0]
            return X, y
    except Exception as exc:
        print(f"Impossibile caricare il dataset preprocessato dal volume: {exc}")

    return None


def apply_basic_preprocessing_to_test(df_test):
    """
    Applica al test la rimozione colonne e l'imputazione usando le info salvate.
    Le trasformazioni aggiuntive (dummies, PCA, scaling) sono gestite dal modello.
    """
    global _current_preprocess_info

    if _current_preprocess_info is None:
        raise ValueError(
            "Nessuna info di preprocessing disponibile. Esegui prima /dataset/preprocess con save_state=true")

    X_test = df_test.copy()

    # Ottieni il metodo di imputazione
    imputation_method = _current_preprocess_info.get('imputation_method', 'statistical')
    final_columns = _current_preprocess_info.get('final_columns', [])

    print(f"Shape prima del preprocessing: {X_test.shape}")
    print(f"Imputation method: {imputation_method}")

    # Feature engineering coerente con train_processed.csv, se quelle colonne sono attese.
    if 'family_size' in final_columns and {'SibSp', 'Parch'}.issubset(X_test.columns):
        X_test['family_size'] = X_test['SibSp'].fillna(0) + X_test['Parch'].fillna(0) + 1

    if 'title' in final_columns and 'Name' in X_test.columns:
        X_test['title'] = X_test['Name'].astype(str).str.extract(r',\s*([^\.]+)\.', expand=False).fillna('Unknown')

    if 'Fare_log' in final_columns and 'Fare' in X_test.columns:
        fare_values = pd.to_numeric(X_test['Fare'], errors='coerce')
        X_test['Fare_log'] = np.log1p(fare_values.clip(lower=0))

    if 'Age_norm' in final_columns and 'Age' in X_test.columns:
        age_values = pd.to_numeric(X_test['Age'], errors='coerce')
        # Min/max del Titanic train classico: evita di normalizzare sul solo test set.
        X_test['Age_norm'] = (age_values - 0.42) / (80.0 - 0.42)

    # 1. Elimina le stesse colonne rimosse durante il training
    columns_removed = _current_preprocess_info.get('columns_removed', [])
    if columns_removed:
        cols_to_remove = [col for col in columns_removed if col in X_test.columns]
        if cols_to_remove:
            X_test = X_test.drop(columns=cols_to_remove)
            print(f"Colonne eliminate dal test: {cols_to_remove}")
            print(f"Shape dopo rimozione colonne: {X_test.shape}")

    # 2. Applica imputazione
    if imputation_method == 'knn':
        # Imputazione KNN usando l'imputer salvato durante il training
        knn_imputer = _current_preprocess_info.get('knn_imputer')
        knn_encoders = _current_preprocess_info.get('knn_encoders', {})
        knn_feature_names = _current_preprocess_info.get('knn_feature_names', [])
        knn_numeric_cols = _current_preprocess_info.get('knn_numeric_cols', [])

        if knn_imputer is None:
            raise ValueError("Imputer KNN non trovato. Assicurati che il training abbia usato KNN.")

        print("Applicando imputazione KNN al test set...")

        # Assicurati che tutte le colonne necessarie siano presenti
        # Aggiungi colonne mancanti con valore 0
        for col in knn_feature_names:
            if col not in X_test.columns:
                X_test[col] = 0
                print(f"Colonna {col} aggiunta con valore 0")

        # Seleziona solo le colonne necessarie nell'ordine corretto
        X_test_aligned = X_test[knn_feature_names].copy()

        # Converti colonne categoriche in numeriche usando gli encoder salvati
        for col, encoder_info in knn_encoders.items():
            if col in X_test_aligned.columns:
                le = encoder_info['encoder']
                default_category = encoder_info.get('default_category', 'unknown')

                def safe_transform(x):
                    try:
                        return le.transform([str(x)])[0]
                    except ValueError:
                        return le.transform([default_category])[0]

                X_test_aligned[col] = X_test_aligned[col].astype(str).apply(safe_transform)

        # Verifica che X_test_aligned non sia vuoto
        if len(X_test_aligned) == 0:
            raise ValueError("Nessuna riga valida nel test set dopo l'allineamento")

        if X_test_aligned.shape[1] == 0:
            raise ValueError("Nessuna colonna valida nel test set dopo l'allineamento")

        print(f"Shape prima della trasformazione KNN: {X_test_aligned.shape}")

        # Applica KNN imputer
        X_test_imputed_array = knn_imputer.transform(X_test_aligned)

        # Ricostruisci DataFrame
        X_test_imputed = pd.DataFrame(
            X_test_imputed_array,
            columns=knn_feature_names,
            index=X_test.index
        )

        # Reconverti colonne categoriche
        for col, encoder_info in knn_encoders.items():
            if col in X_test_imputed.columns:
                le = encoder_info['encoder']
                rounded = np.clip(np.round(X_test_imputed[col]).astype(int), 0, len(le.classes_) - 1)
                X_test_imputed[col] = le.inverse_transform(rounded)

        X_test = X_test_imputed
        print(f"Imputazione KNN completata. Shape: {X_test.shape}")

    else:
        # Imputazione statistica (media/mediana/moda)
        fill_values = _current_preprocess_info.get('fill_values', {})
        if fill_values:
            for col, value in fill_values.items():
                if col in X_test.columns and X_test[col].isna().any():
                    X_test[col] = X_test[col].fillna(value)
                    print(f"Imputazione colonna {col} con valore: {value}")
        else:
            print("Nessun valore di imputazione trovato. Uso fillna con metodo forward fill.")
            X_test = X_test.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # 3. Rimuovi colonne costanti (se presenti nel training)
    constant_cols = _current_preprocess_info.get('constant_columns_removed', [])
    if constant_cols:
        X_test = X_test.drop(columns=[col for col in constant_cols if col in X_test.columns], errors='ignore')
        print(f"Colonne costanti eliminate dal test: {constant_cols}")

    # 4. Assicurati che le colonne siano nello stesso ordine del training
    if final_columns:
        # Aggiungi colonne mancanti con valore 0
        for col in final_columns:
            if col not in X_test.columns:
                X_test[col] = 0
                print(f"Colonna {col} aggiunta con valore 0")

        # Seleziona solo le colonne del training e riordina
        X_test = X_test[final_columns]
        print(f"Shape finale dopo allineamento colonne: {X_test.shape}")

    return X_test

def save_model_to_cache(model_type, model_instance, results, train_metrics, test_metrics, model_params):
    """
    Salva un modello addestrato nella cache globale.

    Parameters:
    -----------
    model_type : str
        Tipo di modello ('logistic', 'svm', 'randomforest', 'xgboost')
    model_instance : object
        Istanza del modello addestrato
    results : dict
        Risultati del training dal metodo train()
    train_metrics : dict
        Metriche sul training set
    test_metrics : dict
        Metriche sul test set
    model_params : dict
        Parametri usati per il training
    """
    global _models

    _models[model_type] = {
        'model': model_instance,
        'results': results,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model_params': model_params,
        'timestamp': str(pd.Timestamp.now()),
        'is_trained': True
    }


def get_model_from_cache(model_type):
    """
    Recupera un modello dalla cache.

    Parameters:
    -----------
    model_type : str
        Tipo di modello da recuperare

    Returns:
    --------
    dict or None: Dizionario del modello o None se non trovato
    """
    global _models

    if model_type in _models and _models[model_type].get('is_trained', False):
        return _models[model_type]
    return None

def convert_to_serializable(obj: object):
    """Converte oggetti numpy in tipi Python serializzabili JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(convert_to_serializable(k)): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    else:
        return obj


def normalize_csv_path(raw_path, default_path):
    path = raw_path or default_path
    return os.path.abspath(os.path.expanduser(str(path).strip().strip('"').strip("'")))

# ==================== ENDPOINT DATASET ====================

@app.route('/dataset/info', methods=['GET'])
def dataset_info():
    """Restituisce informazioni sul dataset"""
    cleaner, info = get_data()
    return jsonify(info)


@app.route('/dataset/na', methods=['GET'])
def dataset_na():
    """Identifica i valori mancanti nel dataset"""
    cleaner, _ = get_data()
    null_values, miss_percent = cleaner.individua_na()
    return jsonify({
        "null_values": null_values.to_dict(),
        "missing_percent": miss_percent.to_dict(),
        "total_missing": int(null_values.sum()),
        "columns_with_missing": int((null_values > 0).sum())
    })



@app.route('/dataset/preprocess', methods=['POST'])
def dataset_preprocess():
    """
    Preprocessa il dataset
    Body JSON (opzionale):
    {
        "remove_columns": ["col1", "col2"],
        "imputation_method": "statistical" | "knn",
        "knn_neighbors": 5,
        "remove_outliers": true,
        "outlier_method": "iqr",
        "outlier_threshold": 1.5,
        "save_state": true
    }
    """
    global _processed_X, _processed_y, _is_processed, _current_preprocess_params, _current_preprocess_info, _test_df, _test_processed

    data = request.get_json() or {}

    # Parametri
    remove_columns = data.get('remove_columns', [])
    imputation_method = data.get('imputation_method', 'statistical')
    knn_neighbors = data.get('knn_neighbors', 5)
    remove_outliers = data.get('remove_outliers', False)
    outlier_method = data.get('outlier_method', 'iqr')
    outlier_threshold = data.get('outlier_threshold', 1.5)
    save_state = data.get('save_state', False)

    _current_preprocess_params = {
        'remove_columns': remove_columns,
        'imputation_method': imputation_method,
        'knn_neighbors': knn_neighbors,
        'remove_outliers': remove_outliers,
        'outlier_method': outlier_method,
        'outlier_threshold': outlier_threshold
    }

    # Carica i dati
    cleaner, info = get_data()

    # 1. Elimina colonne specificate (se presenti)
    columns_dropped = []
    if remove_columns:
        X_temp, y_temp, dropped = cleaner.elimina_colonne(columns_to_drop=remove_columns)
        columns_dropped = dropped
        print(f"Colonne eliminate: {columns_dropped}")

    # 2. Gestione NA
    X, y, preprocess_info, fill_values, knn_imputer, knn_le = cleaner.gestione_na(
        imputation_method=imputation_method,
        knn_neighbors=knn_neighbors
    )

    # 3. Rimozione outliers
    outliers_info = None
    if remove_outliers:
        try:
            X, y, outliers_info = cleaner.trova_outliers(
                method=outlier_method,
                threshold=outlier_threshold,
                imputation_method=imputation_method,
                knn_neighbors=knn_neighbors
            )
        except Exception as e:
            outliers_info = {'error': str(e), 'outliers_removed': 0}

    # 4. Salva lo stato se richiesto
    if save_state:
        _processed_X = X
        _processed_y = y
        _is_processed = True
        _test_df = None
        _test_processed = None
        print("Stato del dataset salvato per chiamate future")

    _current_preprocess_info = {
        'columns_removed': columns_dropped,
        'fill_values': fill_values,
        'knn_imputer': knn_imputer,
        'knn_le': knn_le,
        'knn_encoders': knn_le or {},
        'knn_feature_names': preprocess_info.get('knn_stats', {}).get('feature_names', list(X.columns)),
        'knn_numeric_cols': X.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'imputation_method': imputation_method,
        'final_columns': list(X.columns),
        'final_shape': list(X.shape),
        'constant_columns_removed': preprocess_info.get('constant_columns_removed', []),
        'timestamp': str(pd.Timestamp.now())
    }

    if save_state:
        save_processed_data_to_volume(X, y)
        save_preprocess_info_to_volume(_current_preprocess_info)

    # 5. Prepara risultato
    result = {
        'preprocessing_info': {
            'columns_removed': columns_dropped,
            'n_columns_removed': len(columns_dropped),
            'remaining_columns': list(X.columns) if X is not None else [],
            'imputation_method': imputation_method,
            'knn_neighbors': knn_neighbors if imputation_method == 'knn' else None,
            'remove_outliers': remove_outliers,
            'duplicates_removed': preprocess_info.get('duplicates_removed', 0),
            'constant_columns_removed': preprocess_info.get('constant_columns_removed', 0),
            'final_shape': list(X.shape) if X is not None else None,
            'state_saved': save_state,
            'is_processed': _is_processed
        },
        'data_shape': list(X.shape) if X is not None else None,
        'target_info': {
            'unique_values': int(y.nunique()) if y is not None else None,
            'distribution': y.value_counts().to_dict() if y is not None else None,
            'class_balance': {
                str(k): float(v / len(y)) for k, v in y.value_counts().to_dict().items()
            } if y is not None else None
        }
    }

    if outliers_info:
        result['outliers_info'] = outliers_info

    return jsonify(convert_to_serializable(result))


@app.route('/dataset/get_processed', methods=['GET'])
def get_processed_data():
    """Restituisce i dati preprocessati salvati"""
    global _processed_X, _processed_y, _is_processed

    if not _is_processed:
        return jsonify({
                           'error': 'Nessun dato preprocessato disponibile. Esegui prima /dataset/preprocess con save_state=true'}), 404

    return jsonify({
        'is_processed': True,
        'data_shape': list(_processed_X.shape) if _processed_X is not None else None,
        'columns': list(_processed_X.columns) if _processed_X is not None else [],
        'target_distribution': _processed_y.value_counts().to_dict() if _processed_y is not None else None
    })


@app.route('/dataset/reset', methods=['POST'])
def reset_processed_data():
    """Resetta i dati preprocessati"""
    global _processed_X, _processed_y, _is_processed, _current_preprocess_info

    _processed_X = None
    _processed_y = None
    _is_processed = False
    _current_preprocess_info = None

    for path in (PROCESSED_X_PATH, PROCESSED_Y_PATH, PROCESSED_INFO_PATH):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception as exc:
            print(f"Impossibile eliminare {path}: {exc}")

    return jsonify({'message': 'Stato preprocessato resettato con successo'})


# ==================== ENDPOINT STATISTICHE ====================

@app.route('/stats/descriptive', methods=['GET'])
def stats_descriptive():
    """Statistiche descrittive del dataset"""
    cleaner, info = get_data()
    stats = StatisticaDescrittiva(cleaner)
    return jsonify(stats.statistica_descrittiva())


@app.route('/stats/quick', methods=['GET'])
def stats_quick():
    """Riepilogo rapido del dataset"""
    cleaner, info = get_data()
    stats = StatisticaDescrittiva(cleaner)
    return jsonify(stats.quick_summary())


@app.route('/stats/correlation', methods=['GET'])
def stats_correlation():
    """Matrice di correlazione"""
    metodo = request.args.get('method', 'pearson')
    cleaner, _ = get_data()
    viz = Visualizzatore(cleaner)
    corr_matrix = viz.matrice_correlazione(metodo=metodo)
    return jsonify(corr_matrix)


# ==================== ENDPOINT VISUALIZZAZIONI ====================

@app.route('/plots/distributions', methods=['GET'])
def plot_distributions():
    """Grafici distribuzioni"""
    cleaner, _ = get_data()
    viz = Visualizzatore(cleaner)
    img = viz.grafici_distribuzioni()
    return send_file(img, mimetype='image/png')


@app.route('/plots/boxplot', methods=['GET'])
def plot_boxplot():
    """Grafici boxplot"""
    cleaner, _ = get_data()
    viz = Visualizzatore(cleaner)
    img = viz.grafici_boxplot()
    return send_file(img, mimetype='image/png')


@app.route('/plots/correlation', methods=['GET'])
def plot_correlation():
    """Heatmap di correlazione"""
    metodo = request.args.get('method', 'pearson')
    cleaner, _ = get_data()
    viz = Visualizzatore(cleaner)
    img = viz.grafici_correlazione(metodo=metodo)
    return send_file(img, mimetype='image/png')


@app.route('/plots/pairplot', methods=['GET'])
def plot_pairplot():
    """Pairplot delle variabili"""
    vars_param = request.args.get('vars', None)
    hue = request.args.get('hue', None)
    vars_list = vars_param.split(',') if vars_param else None

    cleaner, _ = get_data()
    viz = Visualizzatore(cleaner)
    img = viz.grafico_pairplot(vars=vars_list, hue=hue)
    return send_file(img, mimetype='image/png')


# ==================== ENDPOINT LOGISTIC REGRESSION ====================

@app.route('/logreg/train', methods=['POST'])
def logistic_train():
    data = request.get_json() or {}
    use_pca = data.get('use_pca', False)
    n_components = data.get('n_components', None)

    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    model = RegressioneLogistica(X, y, use_pca=use_pca, n_components=n_components)
    results = model.train(
        C=data.get('C', 1.0),
        penalty=data.get('penalty', 'l2'),
        max_iter=data.get('max_iter', 1000)
    )

    eval_train = Evaluation(results['y_train'], results['y_pred_train'], classification=True)
    train_metrics = eval_train.metrics()

    eval_test = Evaluation(results['y_test'], results['y_pred_test'], classification=True)
    test_metrics = eval_test.metrics()

    model_params = {
        'C': data.get('C', 1.0),
        'penalty': data.get('penalty', 'l2'),
        'max_iter': data.get('max_iter', 1000),
        'use_pca': use_pca,
        'n_components': n_components
    }

    save_model_to_cache('logistic', model, results, train_metrics, test_metrics, model_params)

    return jsonify(convert_to_serializable({
        'accuracy_train': train_metrics.get('Accuracy'),
        'accuracy_test': test_metrics.get('Accuracy'),
        'precision_train': train_metrics.get('Precision'),
        'precision_test': test_metrics.get('Precision'),
        'recall_train': train_metrics.get('Recall'),
        'recall_test': test_metrics.get('Recall'),
        'f1_train': train_metrics.get('F1-Score'),
        'f1_test': test_metrics.get('F1-Score'),
        'balanced_accuracy_train': train_metrics.get('Balanced Accuracy'),
        'balanced_accuracy_test': test_metrics.get('Balanced Accuracy'),
        'params': results.get('params', {}),
        'use_pca': use_pca
    }))

@app.route('/logreg/cv', methods=['POST'])
def logistic_crossval():
    data = request.get_json() or {}
    k = data.get('k', 5)
    method = data.get('method', 'kfold')

    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    cv = KFoldCrossValidation(
        model_class=RegressioneLogistica,
        model_params={},
        k=k,
        method=method,
        classification=True
    )

    results = cv.validate(X, y, train_params={'max_iter': 1000})
    return jsonify(convert_to_serializable({
        'mean_score': results.get('mean_score'),
        'std_score': results.get('std_score'),
        'scores_per_fold': results.get('scores_per_fold'),
        'min_score': results.get('min_score'),
        'max_score': results.get('max_score'),
        'n_folds': results.get('n_folds'),
        'method': method,
        'k': k
    }))


# ==================== ENDPOINT SVM ====================

@app.route('/svc/train', methods=['POST'])
def svm_train():
    data = request.get_json() or {}
    use_pca = data.get('use_pca', False)
    n_components = data.get('n_components', None)

    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    model = SupportVectorMachine(X, y, use_pca=use_pca, n_components=n_components)
    results = model.train(
        kernel=data.get('kernel', 'rbf'),
        C=data.get('C', 1.0),
        gamma=data.get('gamma', 'scale')
    )

    # Calcola le metriche usando la classe Evaluation
    eval_train = Evaluation(results['y_train'], results['y_pred_train'], classification=True)
    train_metrics = eval_train.metrics()

    eval_test = Evaluation(results['y_test'], results['y_pred_test'], classification=True)
    test_metrics = eval_test.metrics()

    model_params = {
        'kernel': data.get('kernel', 'rbf'),
        'C': data.get('C', 1.0),
        'gamma': data.get('gamma', 'scale'),
        'use_pca': use_pca,
        'n_components': n_components
    }

    save_model_to_cache('svm', model, results, train_metrics, test_metrics, model_params)

    return jsonify(convert_to_serializable({
        'accuracy_train': train_metrics.get('Accuracy'),
        'accuracy_test': test_metrics.get('Accuracy'),
        'precision_train': train_metrics.get('Precision'),
        'precision_test': test_metrics.get('Precision'),
        'recall_train': train_metrics.get('Recall'),
        'recall_test': test_metrics.get('Recall'),
        'f1_train': train_metrics.get('F1-Score'),
        'f1_test': test_metrics.get('F1-Score'),
        'balanced_accuracy_train': train_metrics.get('Balanced Accuracy'),
        'balanced_accuracy_test': test_metrics.get('Balanced Accuracy'),
        'params': results.get('params', {}),
        'use_pca': use_pca
    }))


@app.route('/svc/cv', methods=['POST'])
def svm_crossval():
    data = request.get_json() or {}
    k = data.get('k', 5)
    method = data.get('method', 'kfold')

    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    cv = KFoldCrossValidation(
        model_class=SupportVectorMachine,
        model_params={},
        k=k,
        method=method,
        classification=True
    )

    results = cv.validate(X, y, train_params={'kernel': 'rbf', 'C': 1.0})
    return jsonify(convert_to_serializable({
        'mean_score': results.get('mean_score'),
        'std_score': results.get('std_score'),
        'scores_per_fold': results.get('scores_per_fold'),
        'min_score': results.get('min_score'),
        'max_score': results.get('max_score'),
        'n_folds': results.get('n_folds'),
        'method': method,
        'k': k
    }))


# ==================== ENDPOINT RANDOM FOREST ====================

@app.route('/rf/train', methods=['POST'])
def rf_train():
    data = request.get_json() or {}
    use_pca = data.get('use_pca', False)
    n_components = data.get('n_components', None)

    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    model = RandomForest(X, y, use_pca=use_pca, n_components=n_components)
    results = model.train(
        n_estimators=data.get('n_estimators', 100),
        max_depth=data.get('max_depth', None),
        min_samples_split=data.get('min_samples_split', 2)
    )

    # Calcola le metriche usando la classe Evaluation
    eval_train = Evaluation(results['y_train'], results['y_pred_train'], classification=True)
    train_metrics = eval_train.metrics()

    eval_test = Evaluation(results['y_test'], results['y_pred_test'], classification=True)
    test_metrics = eval_test.metrics()

    model_params = {
        'n_estimators': data.get('n_estimators', 100),
        'max_depth': data.get('max_depth', None),
        'min_samples_split': data.get('min_samples_split', 2),
        'use_pca': use_pca,
        'n_components': n_components
    }

    save_model_to_cache('randomforest', model, results, train_metrics, test_metrics, model_params)

    return jsonify(convert_to_serializable({
        'accuracy_train': train_metrics.get('Accuracy'),
        'accuracy_test': test_metrics.get('Accuracy'),
        'precision_train': train_metrics.get('Precision'),
        'precision_test': test_metrics.get('Precision'),
        'recall_train': train_metrics.get('Recall'),
        'recall_test': test_metrics.get('Recall'),
        'f1_train': train_metrics.get('F1-Score'),
        'f1_test': test_metrics.get('F1-Score'),
        'balanced_accuracy_train': train_metrics.get('Balanced Accuracy'),
        'balanced_accuracy_test': test_metrics.get('Balanced Accuracy'),
        'params': results.get('params', {}),
        'use_pca': use_pca
    }))


@app.route('/rf/cv', methods=['POST'])
def rf_crossval():
    data = request.get_json() or {}
    k = data.get('k', 5)
    method = data.get('method', 'kfold')

    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    cv = KFoldCrossValidation(
        model_class=RandomForest,
        model_params={},
        k=k,
        method=method,
        classification=True
    )

    results = cv.validate(X, y, train_params={'n_estimators': 100})
    return jsonify(convert_to_serializable({
        'mean_score': results.get('mean_score'),
        'std_score': results.get('std_score'),
        'scores_per_fold': results.get('scores_per_fold'),
        'min_score': results.get('min_score'),
        'max_score': results.get('max_score'),
        'n_folds': results.get('n_folds'),
        'method': method,
        'k': k
    }))


# ==================== ENDPOINT XGBOOST ====================

@app.route('/xgb/train', methods=['POST'])
def xgb_train():
    data = request.get_json() or {}
    use_pca = data.get('use_pca', False)
    n_components = data.get('n_components', None)

    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    model = XGBoost(X, y, use_pca=use_pca, n_components=n_components)
    results = model.train(
        n_estimators=data.get('n_estimators', 100),
        max_depth=data.get('max_depth', 3),
        learning_rate=data.get('learning_rate', 0.1)
    )

    # Calcola le metriche usando la classe Evaluation
    eval_train = Evaluation(results['y_train'], results['y_pred_train'], classification=True)
    train_metrics = eval_train.metrics()

    eval_test = Evaluation(results['y_test'], results['y_pred_test'], classification=True)
    test_metrics = eval_test.metrics()

    model_params = {
        'n_estimators': data.get('n_estimators', 100),
        'max_depth': data.get('max_depth', 3),
        'learning_rate': data.get('learning_rate', 0.1),
        'use_pca': use_pca,
        'n_components': n_components
    }

    save_model_to_cache('xgboost', model, results, train_metrics, test_metrics, model_params)

    return jsonify(convert_to_serializable({
        'accuracy_train': train_metrics.get('Accuracy'),
        'accuracy_test': test_metrics.get('Accuracy'),
        'precision_train': train_metrics.get('Precision'),
        'precision_test': test_metrics.get('Precision'),
        'recall_train': train_metrics.get('Recall'),
        'recall_test': test_metrics.get('Recall'),
        'f1_train': train_metrics.get('F1-Score'),
        'f1_test': test_metrics.get('F1-Score'),
        'balanced_accuracy_train': train_metrics.get('Balanced Accuracy'),
        'balanced_accuracy_test': test_metrics.get('Balanced Accuracy'),
        'params': results.get('params', {}),
        'use_pca': use_pca
    }))


@app.route('/xgb/cv', methods=['POST'])
def xgb_crossval():
    data = request.get_json() or {}
    k = data.get('k', 5)
    method = data.get('method', 'kfold')

    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    cv = KFoldCrossValidation(
        model_class=XGBoost,
        model_params={},
        k=k,
        method=method,
        classification=True
    )

    results = cv.validate(X, y, train_params={'n_estimators': 100, 'verbose': 0})
    return jsonify(convert_to_serializable({
        'mean_score': results.get('mean_score'),
        'std_score': results.get('std_score'),
        'scores_per_fold': results.get('scores_per_fold'),
        'min_score': results.get('min_score'),
        'max_score': results.get('max_score'),
        'n_folds': results.get('n_folds'),
        'method': method,
        'k': k
    }))


# ==================== ENDPOINT FINE TUNING ====================

@app.route('/finetuning/grid', methods=['POST'])
def finetuning_grid():
    data = request.get_json()
    if not data or 'model_type' not in data:
        return jsonify({'error': 'Specificare model_type'}), 400

    model_type = data['model_type']
    param_grid = data.get('param_grid', {})
    cv = data.get('cv', 5)

    model_map = {
        'logistic': RegressioneLogistica,
        'svm': SupportVectorMachine,
        'randomforest': RandomForest,
        'xgboost': XGBoost
    }

    if model_type not in model_map:
        return jsonify({'error': f'Model type non supportato. Usa: {list(model_map.keys())}'}), 400

    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    ft = FineTuning(
        model_class=model_map[model_type],
        model_init_params={},
        classification=True,
        cv=cv
    )

    try:
        results = ft.grid_search(X, y, param_grid)
    except Exception as e:
        return jsonify({
            'error': 'Errore durante Grid Search',
            'details': str(e)
        }), 400

    return jsonify(convert_to_serializable({
        'best_params': results['best_params'],
        'best_score': results['best_score'],
        'scoring_metric': results['scoring_metric'],
        'n_candidates': results['n_candidates']
    }))


@app.route('/finetuning/random', methods=['POST'])
def finetuning_random():
    data = request.get_json()
    if not data or 'model_type' not in data:
        return jsonify({'error': 'Specificare model_type'}), 400

    model_type = data['model_type']
    param_dist = data.get('param_distributions', {})
    cv = data.get('cv', 5)
    n_iter = data.get('n_iter', 20)

    model_map = {
        'logistic': RegressioneLogistica,
        'svm': SupportVectorMachine,
        'randomforest': RandomForest,
        'xgboost': XGBoost
    }

    if model_type not in model_map:
        return jsonify({'error': f'Model type non supportato. Usa: {list(model_map.keys())}'}), 400

    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    ft = FineTuning(
        model_class=model_map[model_type],
        model_init_params={},
        classification=True,
        cv=cv,
        n_iter_random=n_iter
    )

    try:
        results = ft.random_search(X, y, param_dist, n_iter=n_iter)
    except Exception as e:
        return jsonify({
            'error': 'Errore durante Random Search',
            'details': str(e)
        }), 400

    return jsonify(convert_to_serializable({
        'best_params': results['best_params'],
        'best_score': results['best_score'],
        'scoring_metric': results['scoring_metric'],
        'n_iterations': results['n_iterations']
    }))


# ==================== ENDPOINT PER GESTIRE I MODELLI SALVATI ====================

@app.route('/models/list', methods=['GET'])
def list_saved_models():
    """Restituisce la lista di tutti i modelli salvati in cache."""
    global _models

    models_info = {}
    for model_type, model_data in _models.items():
        if model_data.get('is_trained', False):
            models_info[model_type] = {
                'is_trained': True,
                'timestamp': model_data.get('timestamp'),
                'train_accuracy': model_data.get('train_metrics', {}).get('Accuracy'),
                'test_accuracy': model_data.get('test_metrics', {}).get('Accuracy'),
                'model_params': model_data.get('model_params', {})
            }

    return jsonify({
        'models': models_info,
        'n_models': len(models_info),
        'has_models': len(models_info) > 0
    })


@app.route('/models/get/<model_type>', methods=['GET'])
def get_saved_model(model_type):
    """Restituisce le informazioni di un modello specifico salvato."""
    global _models

    model_data = get_model_from_cache(model_type)
    if model_data is None:
        return jsonify({'error': f'Modello {model_type} non trovato in cache'}), 404

    return jsonify(convert_to_serializable({
        'model_type': model_type,
        'is_trained': True,
        'timestamp': model_data.get('timestamp'),
        'train_metrics': model_data.get('train_metrics'),
        'test_metrics': model_data.get('test_metrics'),
        'model_params': model_data.get('model_params'),
        'results_available': model_data.get('results') is not None
    }))


@app.route('/models/clear', methods=['POST'])
def clear_models_cache():
    """Cancella tutti i modelli salvati in cache."""
    global _models

    _models = {}
    return jsonify({'message': 'Cache dei modelli svuotata con successo'})


@app.route('/models/select/<model_type>', methods=['POST'])
def select_model_for_prediction(model_type):
    """
    Seleziona un modello per le predizioni future.
    Questo endpoint permette di scegliere quale modello usare per /predict/test.
    """
    global _models, _selected_model

    model_data = get_model_from_cache(model_type)
    if model_data is None:
        return jsonify(
            {'error': f'Modello {model_type} non trovato in cache. Addestralo prima con /{model_type}/train'}), 404

    _selected_model = model_type

    return jsonify({
        'message': f'Modello {model_type} selezionato per le predizioni',
        'model_type': model_type,
        'train_accuracy': model_data.get('train_metrics', {}).get('Accuracy'),
        'test_accuracy': model_data.get('test_metrics', {}).get('Accuracy'),
        'timestamp': model_data.get('timestamp')
    })


@app.route('/predict/test', methods=['POST'])
@app.route('/models/predict/test', methods=['POST'])
def predict_test():
    """
    Esegue predizioni sul file di test usando il modello selezionato.
    Se non c'è un modello selezionato, usa 'logistic' come default.
    """
    global _processed_X, _processed_y, _is_processed, _current_preprocess_info, _models, _selected_model, _test_df, _test_processed

    data = request.get_json() or {}

    # Verifica che ci siano dati di training preprocessati
    if not _is_processed or _processed_X is None:
        return jsonify({
                           'error': 'Nessun dato preprocessato disponibile. Esegui prima /dataset/preprocess con save_state=true'}), 400

    if _current_preprocess_info is None:
        return jsonify(
            {'error': 'Nessuna info di preprocessing. Esegui prima /dataset/preprocess con save_state=true'}), 400

    # Determina quale modello usare
    model_type = data.get('model_type', _selected_model if _selected_model else 'logistic')
    include_probabilities = data.get('include_probabilities', True)
    test_csv_path = normalize_csv_path(data.get('csv_path'), TEST_CONFIG['csv_path'])

    # Recupera il modello dalla cache
    model_data = get_model_from_cache(model_type)
    if model_data is None:
        return jsonify({'error': f'Modello {model_type} non trovato. Addestralo prima con /{model_type}/train'}), 400

    model = model_data['model']

    # Carica il test
    try:
        if not os.path.isfile(test_csv_path):
            return jsonify(convert_to_serializable({
                'error': 'File di test non trovato',
                'configured_path': TEST_CONFIG['csv_path'],
                'checked_path': test_csv_path,
                'current_working_directory': os.getcwd(),
                'data_dir': DATA_DIR,
                'exists': os.path.exists(test_csv_path),
                'is_file': os.path.isfile(test_csv_path)
            })), 404

        # Carica il test. La cache viene riusata solo se il path e' lo stesso.
        cached_path = getattr(predict_test, '_cached_path', None)
        if _test_df is None or cached_path != test_csv_path:
            _test_df = pd.read_csv(test_csv_path)
            _test_processed = None
            predict_test._cached_path = test_csv_path
            print(f"File di test caricato: {test_csv_path}, shape {_test_df.shape}")

        # Applica preprocessing base al test se non è già stato processato
        if _test_processed is None:
            _test_processed = apply_basic_preprocessing_to_test(_test_df)
            print(f"Test preprocessato: shape {_test_processed.shape}")

        # Predizioni
        predictions = model.predict(_test_processed)

        # Probabilità
        probabilities = None
        if include_probabilities and hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(_test_processed)
            except Exception as e:
                print(f"Errore nel calcolo probabilità: {e}")

        # Crea DataFrame con i risultati
        output_df = _test_df.copy()
        output_df['prediction'] = predictions

        if probabilities is not None:
            if len(probabilities.shape) == 2 and probabilities.shape[1] == 2:
                output_df['probability_0'] = probabilities[:, 0]
                output_df['probability_1'] = probabilities[:, 1]
            elif len(probabilities.shape) == 2:
                for i in range(probabilities.shape[1]):
                    output_df[f'probability_class_{i}'] = probabilities[:, i]

        # Restituisci come CSV
        output = StringIO()
        output_df.to_csv(output, index=False)
        output.seek(0)

        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment;filename=predictions_{model_type}.csv'}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Errore durante le predizioni: {str(e)}'}), 500


# ==================== HEALTH CHECK ====================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'service': 'ML API',
        'version': '1.0.0'
    })


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'Machine Learning API',
        'version': '1.0.0',
        'endpoints': {
            'dataset': {
                '/dataset/preprocess': 'POST - Preprocessing e salvataggio',
                '/dataset/get_processed': 'GET - Stato cache',
                '/dataset/reset': 'POST - Reset cache'
            },
            'models': {
                '/logreg/train': 'POST - Logistic Regression',
                '/logreg/cv': 'POST - Cross Validation',
                '/svc/train': 'POST - SVM',
                '/svc/cv': 'POST - Cross Validation',
                '/rf/train': 'POST - Random Forest',
                '/rf/cv': 'POST - Cross Validation',
                '/xgb/train': 'POST - XGBoost',
                '/xgb/cv': 'POST - Cross Validation'
            },
            'models_cache': {
                '/models/list': 'GET - Lista modelli salvati',
                '/models/get/<model_type>': 'GET - Info modello specifico',
                '/models/clear': 'POST - Svuota cache modelli',
                '/models/select/<model_type>': 'POST - Seleziona modello per predizioni'
            },
            'predictions': {
                '/predict/test': 'POST - Predizioni su file di test (usa modello selezionato)',
                '/predict/test': 'POST - Predizioni su file di test'
            },
            'finetuning': {
                '/finetuning/grid': 'POST - Grid Search',
                '/finetuning/random': 'POST - Random Search'
            }
        }
    })


