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


app = Flask(__name__)

# Configurazione globale
DATASET_CONFIG = {
    'source': 'csv',  # 'uci' o 'csv'
    'csv_path': r"C:\Users\alisi\Downloads\train.csv",  # Modifica con il percorso del tuo dataset
    'target_column': 'Survived'  # Modifica con il nome della colonna target
}


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


def get_processed_data(imputation_method='statistical', knn_neighbors=5, remove_outliers=False):
    """Carica e processa i dati"""
    cleaner, info = get_data()
    X, y, preprocessing_info = cleaner.gestione_na(
        imputation_method=imputation_method,
        knn_neighbors=knn_neighbors
    )

    if remove_outliers:
        X, y, outliers_info = cleaner.trova_outliers(
            method='iqr',
            threshold=1.5,
            imputation_method=imputation_method,
            knn_neighbors=knn_neighbors
        )
    else:
        outliers_info = None

    return X, y, cleaner, preprocessing_info, outliers_info


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
        "imputation_method": "statistical" | "knn",
        "knn_neighbors": 5,
        "remove_outliers": true,
        "outlier_method": "iqr",
        "outlier_threshold": 1.5
    }
    """
    data = request.get_json() or {}
    imputation_method = data.get('imputation_method', 'statistical')
    knn_neighbors = data.get('knn_neighbors', 5)
    remove_outliers = data.get('remove_outliers', False)

    X, y, cleaner, preprocess_info, outliers_info = get_processed_data(
        imputation_method=imputation_method,
        knn_neighbors=knn_neighbors,
        remove_outliers=remove_outliers
    )

    # Converti in formato serializzabile
    result = {
        'preprocessing_info': preprocess_info,
        'data_shape': list(X.shape) if X is not None else None,
        'target_info': {
            'unique_values': int(y.nunique()) if y is not None else None,
            'distribution': y.value_counts().to_dict() if y is not None else None
        }
    }

    if outliers_info:
        result['outliers_info'] = outliers_info

    return jsonify(result)


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
    stats = StatisticaDescriptiva(cleaner)
    return jsonify(stats.statistica_descrittiva())


@app.route('/stats/quick', methods=['GET'])
def stats_quick():
    """Riepilogo rapido del dataset"""
    cleaner, info = get_data()
    stats = StatisticaDescriptiva(cleaner)
    return jsonify(stats.quick_summary())


@app.route('/stats/correlation', methods=['GET'])
def stats_correlation():
    """Matrice di correlazione"""
    metodo = request.args.get('method', 'pearson')
    cleaner, _ = get_data()
    stats = StatisticaDescriptiva(cleaner)
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
    """Addestra modello Logistic Regression"""
    data = request.get_json() or {}
    imputation_method = data.get('imputation_method', 'statistical')
    use_pca = data.get('use_pca', False)
    n_components = data.get('n_components', None)

    X, y, _, _, _ = get_processed_data(imputation_method=imputation_method)

    model = RegressioneLogistica(X, y, use_pca=use_pca, n_components=n_components)
    results = model.train(
        C=data.get('C', 1.0),
        penalty=data.get('penalty', 'l2'),
        max_iter=data.get('max_iter', 1000)
    )

    return jsonify({
        'accuracy_train': float(results['accuracy_train']) if 'accuracy_train' in results else None,
        'accuracy_test': float(results['accuracy_test']) if 'accuracy_test' in results else None,
        'params': results.get('params', {}),
        'use_pca': use_pca
    })


@app.route('/logreg/coeff', methods=['GET'])
def logistic_coeff():
    """Grafico coefficienti Logistic Regression"""
    imputation_method = request.args.get('imputation_method', 'statistical')
    X, y, _, _, _ = get_processed_data(imputation_method=imputation_method)

    model = RegressioneLogistica(X, y)
    results = model.train()
    img = model.grafico_coefficienti()
    return send_file(img, mimetype='image/png')


@app.route('/logreg/summary', methods=['GET'])
def logistic_summary():
    """Summary Logistic Regression"""
    imputation_method = request.args.get('imputation_method', 'statistical')
    X, y, _, _, _ = get_processed_data(imputation_method=imputation_method)

    model = RegressioneLogistica(X, y)
    results = model.train()

    summary = {
        'model_type': 'Logistic Regression',
        'accuracy_train': float(results['accuracy_train']) if 'accuracy_train' in results else None,
        'accuracy_test': float(results['accuracy_test']) if 'accuracy_test' in results else None,
        'coefficients': results['coefficienti'].to_dict() if results.get('coefficienti') is not None else None,
        'intercept': float(results['intercept']) if results.get('intercept') is not None else None
    }
    return jsonify(summary)


@app.route('/logreg/cv', methods=['POST'])
def logistic_crossval():
    """Cross-validation per Logistic Regression"""
    data = request.get_json() or {}
    imputation_method = data.get('imputation_method', 'statistical')
    k = data.get('k', 5)
    method = data.get('method', 'kfold')

    X, y, _, _, _ = get_processed_data(imputation_method=imputation_method)

    cv = KFoldCrossValidation(
        model_class=RegressioneLogistica,
        model_params={},
        k=k,
        method=method,
        classification=True
    )

    results = cv.validate(X, y, train_params={'max_iter': 1000})
    return jsonify({
        'mean_score': results['mean_score'],
        'std_score': results['std_score'],
        'scores_per_fold': results['scores_per_fold'],
        'method': method,
        'k': k
    })


# ==================== ENDPOINT SVM ====================

@app.route('/svc/train', methods=['POST'])
def svm_train():
    """Addestra modello SVM"""
    data = request.get_json() or {}
    imputation_method = data.get('imputation_method', 'statistical')
    use_pca = data.get('use_pca', False)
    n_components = data.get('n_components', None)

    X, y, _, _, _ = get_processed_data(imputation_method=imputation_method)

    model = SupportVectorMachine(X, y, use_pca=use_pca, n_components=n_components)
    results = model.train(
        kernel=data.get('kernel', 'rbf'),
        C=data.get('C', 1.0),
        gamma=data.get('gamma', 'scale')
    )

    return jsonify({
        'accuracy_train': float(results['accuracy_train']) if 'accuracy_train' in results else None,
        'accuracy_test': float(results['accuracy_test']) if 'accuracy_test' in results else None,
        'params': results.get('params', {}),
        'use_pca': use_pca
    })


@app.route('/svc/summary', methods=['GET'])
def svm_summary():
    """Summary SVM"""
    imputation_method = request.args.get('imputation_method', 'statistical')
    X, y, _, _, _ = get_processed_data(imputation_method=imputation_method)

    model = SupportVectorMachine(X, y)
    results = model.train()

    summary = {
        'model_type': 'Support Vector Machine',
        'accuracy_train': float(results['accuracy_train']) if 'accuracy_train' in results else None,
        'accuracy_test': float(results['accuracy_test']) if 'accuracy_test' in results else None,
        'kernel': results.get('params', {}).get('kernel'),
        'C': results.get('params', {}).get('C')
    }
    return jsonify(summary)


@app.route('/svc/cv', methods=['POST'])
def svm_crossval():
    """Cross-validation per SVM"""
    data = request.get_json() or {}
    imputation_method = data.get('imputation_method', 'statistical')
    k = data.get('k', 5)
    method = data.get('method', 'kfold')

    X, y, _, _, _ = get_processed_data(imputation_method=imputation_method)

    cv = KFoldCrossValidation(
        model_class=SupportVectorMachine,
        model_params={},
        k=k,
        method=method,
        classification=True
    )

    results = cv.validate(X, y, train_params={'kernel': 'rbf', 'C': 1.0})
    return jsonify({
        'mean_score': results['mean_score'],
        'std_score': results['std_score'],
        'scores_per_fold': results['scores_per_fold'],
        'method': method,
        'k': k
    })


# ==================== ENDPOINT RANDOM FOREST ====================

@app.route('/rf/train', methods=['POST'])
def rf_train():
    """Addestra modello Random Forest"""
    data = request.get_json() or {}
    imputation_method = data.get('imputation_method', 'statistical')
    use_pca = data.get('use_pca', False)
    n_components = data.get('n_components', None)

    X, y, _, _, _ = get_processed_data(imputation_method=imputation_method)

    model = RandomForest(X, y, use_pca=use_pca, n_components=n_components)
    results = model.train(
        n_estimators=data.get('n_estimators', 100),
        max_depth=data.get('max_depth', None),
        min_samples_split=data.get('min_samples_split', 2)
    )

    return jsonify({
        'accuracy_train': float(results['accuracy_train']) if 'accuracy_train' in results else None,
        'accuracy_test': float(results['accuracy_test']) if 'accuracy_test' in results else None,
        'params': results.get('params', {}),
        'use_pca': use_pca
    })


@app.route('/rf/importance', methods=['GET'])
def rf_importance():
    """Feature importance Random Forest"""
    imputation_method = request.args.get('imputation_method', 'statistical')
    X, y, _, _, _ = get_processed_data(imputation_method=imputation_method)

    model = RandomForest(X, y)
    results = model.train()

    if results.get('feature_importance') is not None:
        importance = results['feature_importance'].to_dict(orient='records')
    else:
        importance = None

    return jsonify({
        'feature_importance': importance,
        'top_features': importance[:10] if importance else None
    })


# ==================== ENDPOINT XGBOOST ====================

@app.route('/xgb/train', methods=['POST'])
def xgb_train():
    """Addestra modello XGBoost"""
    data = request.get_json() or {}
    imputation_method = data.get('imputation_method', 'statistical')
    use_pca = data.get('use_pca', False)
    n_components = data.get('n_components', None)

    X, y, _, _, _ = get_processed_data(imputation_method=imputation_method)

    model = XGBoost(X, y, use_pca=use_pca, n_components=n_components)
    results = model.train(
        n_estimators=data.get('n_estimators', 100),
        max_depth=data.get('max_depth', 3),
        learning_rate=data.get('learning_rate', 0.1)
    )

    return jsonify({
        'accuracy_train': float(results['accuracy_train']) if 'accuracy_train' in results else None,
        'accuracy_test': float(results['accuracy_test']) if 'accuracy_test' in results else None,
        'params': results.get('params', {}),
        'use_pca': use_pca
    })


# ==================== ENDPOINT FINE TUNING ====================

@app.route('/finetuning/grid', methods=['POST'])
def finetuning_grid():
    """Grid Search per ottimizzazione iperparametri"""
    data = request.get_json()
    if not data or 'model_type' not in data:
        return jsonify({'error': 'Specificare model_type'}), 400

    model_type = data['model_type']
    param_grid = data.get('param_grid', {})
    imputation_method = data.get('imputation_method', 'statistical')
    cv = data.get('cv', 5)

    model_map = {
        'logistic': RegressioneLogistica,
        'svm': SupportVectorMachine,
        'randomforest': RandomForest,
        'xgboost': XGBoost
    }

    if model_type not in model_map:
        return jsonify({'error': f'Model type non supportato. Usa: {list(model_map.keys())}'}), 400

    X, y, _, _, _ = get_processed_data(imputation_method=imputation_method)

    ft = FineTuning(
        model_class=model_map[model_type],
        model_init_params={},
        classification=True,
        cv=cv
    )

    results = ft.grid_search(X, y, param_grid)

    return jsonify({
        'best_params': results['best_params'],
        'best_score': results['best_score'],
        'scoring_metric': results['scoring_metric'],
        'n_candidates': results['n_candidates']
    })


@app.route('/finetuning/random', methods=['POST'])
def finetuning_random():
    """Random Search per ottimizzazione iperparametri"""
    data = request.get_json()
    if not data or 'model_type' not in data:
        return jsonify({'error': 'Specificare model_type'}), 400

    model_type = data['model_type']
    param_dist = data.get('param_distributions', {})
    imputation_method = data.get('imputation_method', 'statistical')
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

    X, y, _, _, _ = get_processed_data(imputation_method=imputation_method)

    ft = FineTuning(
        model_class=model_map[model_type],
        model_init_params={},
        classification=True,
        cv=cv,
        n_iter_random=n_iter
    )

    results = ft.random_search(X, y, param_dist, n_iter=n_iter)

    return jsonify({
        'best_params': results['best_params'],
        'best_score': results['best_score'],
        'scoring_metric': results['scoring_metric'],
        'n_iterations': results['n_iterations']
    })


# ==================== HEALTH CHECK ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check dell'API"""
    return jsonify({
        'status': 'ok',
        'service': 'ML API',
        'version': '1.0.0'
    })


@app.route('/', methods=['GET'])
def index():
    """Endpoint principale con documentazione"""
    return jsonify({
        'service': 'Machine Learning API',
        'endpoints': {
            'dataset': {
                '/dataset/info': 'GET - Info dataset',
                '/dataset/na': 'GET - Valori mancanti',
                '/dataset/preprocess': 'POST - Preprocessing',
                '/dataset/columns': 'DELETE - Elimina colonne'
            },
            'stats': {
                '/stats/descriptive': 'GET - Statistiche descrittive',
                '/stats/quick': 'GET - Riepilogo rapido',
                '/stats/correlation': 'GET - Matrice correlazione'
            },
            'plots': {
                '/plots/distributions': 'GET - Grafici distribuzioni',
                '/plots/boxplot': 'GET - Boxplot',
                '/plots/correlation': 'GET - Heatmap correlazione',
                '/plots/pairplot': 'GET - Pairplot'
            },
            'models': {
                '/logreg/train': 'POST - Logistic Regression',
                '/svc/train': 'POST - SVM',
                '/rf/train': 'POST - Random Forest',
                '/xgb/train': 'POST - XGBoost'
            },
            'cross_validation': {
                '/logreg/cv': 'POST - CV Logistic Regression',
                '/svc/cv': 'POST - CV SVM'
            },
            'finetuning': {
                '/finetuning/grid': 'POST - Grid Search',
                '/finetuning/random': 'POST - Random Search'
            }
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)