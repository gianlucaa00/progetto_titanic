from typing import Any
import numpy as np
import pandas as pd
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
DATASET_CONFIG = {
    'source': 'csv',  # 'uci' o 'csv'
    'csv_path': r"C:\Users\alisi\Downloads\train.csv",  # Modifica con il percorso del tuo dataset
    'target_column': 'Survived'  # Modifica con il nome della colonna target
}
#TEST_CONFIG = {
#    'source': 'csv',  # 'uci' o 'csv'
#    'csv_path': r"C:\Users\alisi\Downloads\test.csv",  # Modifica con il percorso del tuo dataset
#}

# Variabili globali per salvare lo stato
_processed_X = None
_processed_y = None
_is_processed = False
_current_preprocess_params = {}



def get_data():
    """Carica i dati e restituisce l'istanza di CleanDataset"""
    dataset = Dataset(
        source=DATASET_CONFIG['source'],
        csv_path=DATASET_CONFIG['csv_path'],
        target_column=DATASET_CONFIG['target_column']
    )
    X, y = dataset.get_data()
    cleaner = CleanDataset(X, y)
    return cleaner, dataset.get_info()


def get_processed_data_cached(use_cache=True):
    """
    Restituisce i dati preprocessati.
    Se use_cache=True e i dati sono stati salvati, usa quelli.
    """
    global _processed_X, _processed_y, _is_processed

    if _is_processed and _processed_X is not None:
        print("Usando dati preprocessati dalla cache")
        return _processed_X, _processed_y
    else:
        return None, None

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
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    else:
        return obj

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
    global _processed_X, _processed_y, _is_processed

    data = request.get_json() or {}

    # Parametri
    remove_columns = data.get('remove_columns', [])
    imputation_method = data.get('imputation_method', 'statistical')
    knn_neighbors = data.get('knn_neighbors', 5)
    remove_outliers = data.get('remove_outliers', False)
    outlier_method = data.get('outlier_method', 'iqr')
    outlier_threshold = data.get('outlier_threshold', 1.5)
    save_state = data.get('save_state', False)  # Nuovo parametro

    # Carica i dati
    cleaner, info = get_data()

    # 1. Elimina colonne specificate (se presenti)
    columns_dropped = []
    if remove_columns:
        X_temp, y_temp, dropped = cleaner.elimina_colonne(columns_to_drop=remove_columns)
        columns_dropped = dropped
        print(f"Colonne eliminate: {columns_dropped}")

    # 2. Gestione NA
    X, y, preprocess_info = cleaner.gestione_na(
        imputation_method=imputation_method,
        knn_neighbors=knn_neighbors
    )

    # 3. Rimozione outliers (opzionale)
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
        print("Stato del dataset salvato per chiamate future")

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

    return jsonify(result)


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
    global _processed_X, _processed_y, _is_processed

    _processed_X = None
    _processed_y = None
    _is_processed = False

    return jsonify({'message': 'Stato preprocessato resettato con successo'})


@app.route('/dataset/columns', methods=['DELETE'])
def delete_columns():
    """Elimina colonne specificate
    Body JSON:
    {
        "columns": ["col1", "col2"]
    }
    """
    data = request.get_json()
    if not data or 'columns' not in data:
        return jsonify({'error': 'Specificare la lista delle colonne da eliminare'}), 400

    cleaner, _ = get_data()
    columns_to_drop = data['columns']
    X, y, dropped = cleaner.elimina_colonne(columns_to_drop=columns_to_drop)

    return jsonify({
        'columns_dropped': dropped,
        'remaining_columns': list(X.columns) if X is not None else [],
        'new_shape': list(X.shape) if X is not None else None
    })


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
    stats = StatisticaDescrittiva(cleaner)
    corr_matrix = stats.matrice_correlazione(metodo=metodo)
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


@app.route('/logreg/summary', methods=['GET'])
def logistic_summary():
    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    model = RegressioneLogistica(X, y)
    results = model.train()

    # Calcola le metriche
    eval_train = Evaluation(results['y_train'], results['y_pred_train'], classification=True)
    train_metrics = eval_train.metrics()

    eval_test = Evaluation(results['y_test'], results['y_pred_test'], classification=True)
    test_metrics = eval_test.metrics()

    coefficients = None
    if results.get('coefficienti') is not None:
        coefficients = results['coefficienti'].to_dict(orient='records')

    return jsonify(convert_to_serializable({
        'model_type': 'Logistic Regression',
        'train_metrics': {
            'accuracy': train_metrics.get('Accuracy'),
            'precision': train_metrics.get('Precision'),
            'recall': train_metrics.get('Recall'),
            'f1_score': train_metrics.get('F1-Score'),
            'balanced_accuracy': train_metrics.get('Balanced Accuracy')
        },
        'test_metrics': {
            'accuracy': test_metrics.get('Accuracy'),
            'precision': test_metrics.get('Precision'),
            'recall': test_metrics.get('Recall'),
            'f1_score': test_metrics.get('F1-Score'),
            'balanced_accuracy': test_metrics.get('Balanced Accuracy')
        },
        'coefficients': coefficients,
        'intercept': results.get('intercept'),
        'confusion_matrix': test_metrics.get('Confusion Matrix').tolist() if test_metrics.get(
            'Confusion Matrix') is not None else None
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


@app.route('/svc/summary', methods=['GET'])
def svm_summary():
    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    model = SupportVectorMachine(X, y)
    results = model.train()

    # Calcola le metriche
    eval_train = Evaluation(results['y_train'], results['y_pred_train'], classification=True)
    train_metrics = eval_train.metrics()

    eval_test = Evaluation(results['y_test'], results['y_pred_test'], classification=True)
    test_metrics = eval_test.metrics()

    coefficients = None
    if results.get('coefficienti') is not None:
        coefficients = results['coefficienti'].to_dict(orient='records')

    return jsonify(convert_to_serializable({
        'model_type': 'Support Vector Machine',
        'train_metrics': {
            'accuracy': train_metrics.get('Accuracy'),
            'precision': train_metrics.get('Precision'),
            'recall': train_metrics.get('Recall'),
            'f1_score': train_metrics.get('F1-Score'),
            'balanced_accuracy': train_metrics.get('Balanced Accuracy')
        },
        'test_metrics': {
            'accuracy': test_metrics.get('Accuracy'),
            'precision': test_metrics.get('Precision'),
            'recall': test_metrics.get('Recall'),
            'f1_score': test_metrics.get('F1-Score'),
            'balanced_accuracy': test_metrics.get('Balanced Accuracy')
        },
        'coefficients': coefficients,
        'intercept': results.get('intercept'),
        'confusion_matrix': test_metrics.get('Confusion Matrix').tolist() if test_metrics.get(
            'Confusion Matrix') is not None else None
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


@app.route('/rf/summary', methods=['GET'])
def rf_summary():
    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    model = RandomForest(X, y)
    results = model.train()

    # Calcola le metriche
    eval_train = Evaluation(results['y_train'], results['y_pred_train'], classification=True)
    train_metrics = eval_train.metrics()

    eval_test = Evaluation(results['y_test'], results['y_pred_test'], classification=True)
    test_metrics = eval_test.metrics()

    coefficients = None
    if results.get('coefficienti') is not None:
        coefficients = results['coefficienti'].to_dict(orient='records')

    return jsonify(convert_to_serializable({
        'model_type': 'Random Forest',
        'train_metrics': {
            'accuracy': train_metrics.get('Accuracy'),
            'precision': train_metrics.get('Precision'),
            'recall': train_metrics.get('Recall'),
            'f1_score': train_metrics.get('F1-Score'),
            'balanced_accuracy': train_metrics.get('Balanced Accuracy')
        },
        'test_metrics': {
            'accuracy': test_metrics.get('Accuracy'),
            'precision': test_metrics.get('Precision'),
            'recall': test_metrics.get('Recall'),
            'f1_score': test_metrics.get('F1-Score'),
            'balanced_accuracy': test_metrics.get('Balanced Accuracy')
        },
        'coefficients': coefficients,
        'intercept': results.get('intercept'),
        'confusion_matrix': test_metrics.get('Confusion Matrix').tolist() if test_metrics.get(
            'Confusion Matrix') is not None else None
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

@app.route('/xgb/summary', methods=['GET'])
def xgb_summary():
    X, y = get_processed_data_cached()
    if X is None:
        return jsonify({'error': 'Nessun dato preprocessato disponibile'}), 400

    model = XGBoost(X, y)
    results = model.train()

    # Calcola le metriche
    eval_train = Evaluation(results['y_train'], results['y_pred_train'], classification=True)
    train_metrics = eval_train.metrics()

    eval_test = Evaluation(results['y_test'], results['y_pred_test'], classification=True)
    test_metrics = eval_test.metrics()

    coefficients = None
    if results.get('coefficienti') is not None:
        coefficients = results['coefficienti'].to_dict(orient='records')

    return jsonify(convert_to_serializable({
        'model_type': 'XGBoost',
        'train_metrics': {
            'accuracy': train_metrics.get('Accuracy'),
            'precision': train_metrics.get('Precision'),
            'recall': train_metrics.get('Recall'),
            'f1_score': train_metrics.get('F1-Score'),
            'balanced_accuracy': train_metrics.get('Balanced Accuracy')
        },
        'test_metrics': {
            'accuracy': test_metrics.get('Accuracy'),
            'precision': test_metrics.get('Precision'),
            'recall': test_metrics.get('Recall'),
            'f1_score': test_metrics.get('F1-Score'),
            'balanced_accuracy': test_metrics.get('Balanced Accuracy')
        },
        'coefficients': coefficients,
        'intercept': results.get('intercept'),
        'confusion_matrix': test_metrics.get('Confusion Matrix').tolist() if test_metrics.get(
            'Confusion Matrix') is not None else None
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

    results = ft.grid_search(X, y, param_grid)

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

    results = ft.random_search(X, y, param_dist, n_iter=n_iter)

    return jsonify(convert_to_serializable({
        'best_params': results['best_params'],
        'best_score': results['best_score'],
        'scoring_metric': results['scoring_metric'],
        'n_iterations': results['n_iterations']
    }))


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
        'endpoints': {
            'dataset': {
                '/dataset/preprocess': 'POST - Preprocessing e salvataggio',
                '/dataset/status': 'GET - Stato cache',
                '/dataset/reset': 'POST - Reset cache'
            },
            'models': {
                '/logreg/train': 'POST - Logistic Regression',
                '/logreg/summary': 'GET - Summary Logistic Regression',
                '/logreg/cv': 'POST - CV Logistic Regression',
                '/svc/train': 'POST - SVM',
                '/svc/summary': 'GET - Summary SVM',
                '/svc/cv': 'POST - CV SVM',
                '/rf/train': 'POST - Random Forest',
                '/rf/summary': 'GET - Summary Random Forest',
                '/rf/cv': 'POST - CV Random Forest',
                '/xgb/train': 'POST - XGBoost',
                '/xgb/summary': 'GET - Summary XGBoost',
                '/xgb/cv': 'POST - CV XGBoost'
            },
            'finetuning': {
                '/finetuning/grid': 'POST - Grid Search',
                '/finetuning/random': 'POST - Random Search'
            }
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)