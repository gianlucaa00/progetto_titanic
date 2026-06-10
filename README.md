# Titanic ML API

API Flask per esplorare, preprocessare e modellare il dataset Titanic con diversi algoritmi di machine learning.

Il progetto include caricamento dati, analisi descrittiva, visualizzazioni, preprocessing, training di modelli di classificazione, cross-validation, fine tuning e generazione di predizioni su un file di test.

## Funzionalita principali

- Caricamento dataset da CSV o da UCI ML Repository.
- Analisi dei valori mancanti e statistiche descrittive.
- Preprocessing con:
  - rimozione colonne selezionate;
  - sostituzione valori mancanti "strani";
  - imputazione statistica o KNN;
  - rimozione colonne con troppi valori mancanti;
  - rimozione duplicati;
  - rimozione outlier opzionale.
- Modelli disponibili:
  - Logistic Regression;
  - Support Vector Machine;
  - Random Forest;
  - XGBoost.
- Metriche di classificazione:
  - accuracy;
  - precision;
  - recall;
  - F1-score;
  - balanced accuracy;
  - confusion matrix.
- Cache in memoria per dataset preprocessato e modelli addestrati.
- Predizione sul file `test.csv` e output CSV scaricabile.

## Struttura del progetto

```text
titanic_proj/
+-- README.md
+-- it/
    +-- akron_valtellina/
        +-- app/
        |   +-- api.py
        +-- data/
        |   +-- descriptive_statistics.py
        |   +-- graphs.py
        |   +-- load_data.py
        |   +-- preprocessing.py
        +-- models/
            +-- dataset_split.py
            +-- evaluation.py
            +-- finetuning_iperparametri.py
            +-- kfold_crossvalidation.py
            +-- scaling_pca.py
            +-- classification/
            |   +-- logistic_regression.py
            |   +-- random_forest.py
            |   +-- svc.py
            |   +-- xgboost.py
            +-- regression/
                +-- linear_regression.py
```

## Requisiti

- Python 3.11 consigliato.
- Dataset Titanic in formato CSV.
- Ambiente virtuale Python.

Dipendenze principali:

```text
flask
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
xgboost
ucimlrepo
```

## Installazione

Clona il repository:

```bash
git clone <repository-url>
cd titanic_proj
```

Crea e attiva un ambiente virtuale:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Installa le dipendenze:

```bash
pip install flask pandas numpy scikit-learn scipy matplotlib seaborn xgboost ucimlrepo
```

## Configurazione dataset

Nel file `it/akron_valtellina/app/api.py` sono configurati i percorsi dei dataset:

```python
DATASET_CONFIG = {
    "source": "csv",
    "csv_path": r"C:\Users\alisi\Downloads\train.csv",
    "target_column": "Survived"
}

TEST_CONFIG = {
    "source": "csv",
    "csv_path": r"C:\Users\alisi\Downloads\test.csv"
}
```

Per usare il progetto su un altro computer, modifica `csv_path` con i percorsi locali dei file `train.csv` e `test.csv`.

## Avvio API

Dalla root del progetto:

```bash
set PYTHONPATH=%cd%
python -m it.akron_valtellina.app.api
```

L'API sara disponibile su:

```text
http://localhost:5000
```

Health check:

```bash
curl http://localhost:5000/health
```

Risposta attesa:

```json
{
  "service": "ML API",
  "status": "ok",
  "version": "1.0.0"
}
```

## Flusso consigliato

### 1. Controllare il dataset

```bash
curl http://localhost:5000/dataset/info
```

```bash
curl http://localhost:5000/dataset/na
```

### 2. Preprocessare e salvare lo stato

Per Titanic e' consigliato rimuovere colonne molto identificative o troppo sparse:

```bash
curl -X POST http://localhost:5000/dataset/preprocess ^
  -H "Content-Type: application/json" ^
  -d "{\"save_state\": true, \"remove_columns\": [\"Name\", \"Ticket\", \"Cabin\", \"PassengerId\"]}"
```

Questa chiamata salva il dataset preprocessato nella cache in memoria. Le chiamate di training richiedono che `save_state` sia `true`.

### 3. Verificare lo stato preprocessato

```bash
curl http://localhost:5000/dataset/get_processed
```

### 4. Addestrare un modello

Logistic Regression:

```bash
curl -X POST http://localhost:5000/logreg/train ^
  -H "Content-Type: application/json" ^
  -d "{\"max_iter\": 2000}"
```

Random Forest:

```bash
curl -X POST http://localhost:5000/rf/train ^
  -H "Content-Type: application/json" ^
  -d "{\"n_estimators\": 100, \"max_depth\": 5}"
```

SVM:

```bash
curl -X POST http://localhost:5000/svc/train ^
  -H "Content-Type: application/json" ^
  -d "{\"kernel\": \"rbf\", \"C\": 1.0}"
```

XGBoost:

```bash
curl -X POST http://localhost:5000/xgb/train ^
  -H "Content-Type: application/json" ^
  -d "{\"n_estimators\": 100, \"max_depth\": 3, \"learning_rate\": 0.1}"
```

### 5. Controllare i modelli salvati

```bash
curl http://localhost:5000/models/list
```

### 6. Selezionare un modello per le predizioni

```bash
curl -X POST http://localhost:5000/models/select/logistic
```

Valori supportati:

```text
logistic
svm
randomforest
xgboost
```

### 7. Generare predizioni sul test set

```bash
curl -X POST http://localhost:5000/predict/test ^
  -H "Content-Type: application/json" ^
  -d "{\"model_type\": \"logistic\", \"include_probabilities\": true}" ^
  -o predictions_logistic.csv
```

L'output e' un file CSV con le colonne originali del test set, la colonna `prediction` e, se disponibili, le probabilita di classe.

## Endpoint principali

### Dataset

| Metodo | Endpoint | Descrizione |
| --- | --- | --- |
| GET | `/dataset/info` | Informazioni generali sul dataset |
| GET | `/dataset/na` | Valori mancanti per colonna |
| POST | `/dataset/preprocess` | Preprocessing e salvataggio opzionale in cache |
| GET | `/dataset/get_processed` | Stato del dataset preprocessato |
| POST | `/dataset/reset` | Reset della cache del dataset |

### Statistiche e grafici

| Metodo | Endpoint | Descrizione |
| --- | --- | --- |
| GET | `/stats/quick` | Riepilogo rapido |
| GET | `/stats/descriptive` | Statistiche descrittive complete |
| GET | `/stats/correlation` | Matrice di correlazione |
| GET | `/plots/distributions` | Grafici distribuzioni |
| GET | `/plots/boxplot` | Boxplot numerici |
| GET | `/plots/correlation` | Heatmap correlazione |
| GET | `/plots/pairplot` | Pairplot |

### Modelli

| Metodo | Endpoint | Descrizione |
| --- | --- | --- |
| POST | `/logreg/train` | Training Logistic Regression |
| POST | `/logreg/cv` | Cross-validation Logistic Regression |
| POST | `/svc/train` | Training SVM |
| POST | `/svc/cv` | Cross-validation SVM |
| POST | `/rf/train` | Training Random Forest |
| POST | `/rf/cv` | Cross-validation Random Forest |
| POST | `/xgb/train` | Training XGBoost |
| POST | `/xgb/cv` | Cross-validation XGBoost |

### Fine tuning

| Metodo | Endpoint | Descrizione |
| --- | --- | --- |
| POST | `/finetuning/grid` | Grid Search |
| POST | `/finetuning/random` | Random Search |

### Cache modelli e predizione

| Metodo | Endpoint | Descrizione |
| --- | --- | --- |
| GET | `/models/list` | Lista modelli addestrati in memoria |
| GET | `/models/get/<model_type>` | Dettagli di un modello |
| POST | `/models/select/<model_type>` | Seleziona modello per predizione |
| POST | `/models/clear` | Svuota cache modelli |
| POST | `/predict/test` | Predizioni sul file di test configurato |

## Esempio di risposta training

```json
{
  "accuracy_train": 0.7708,
  "accuracy_test": 0.8039,
  "precision_train": 0.7698,
  "precision_test": 0.8030,
  "recall_train": 0.7708,
  "recall_test": 0.8039,
  "f1_train": 0.7701,
  "f1_test": 0.8029,
  "balanced_accuracy_train": 0.7613,
  "balanced_accuracy_test": 0.7941,
  "use_pca": false
}
```

I valori possono cambiare in base al preprocessing, ai parametri e alla versione delle librerie.

## Note tecniche

- La cache e' solo in memoria: se il server viene riavviato, dataset preprocessato e modelli addestrati vengono persi.
- Prima di addestrare un modello bisogna chiamare `/dataset/preprocess` con `save_state: true`.
- Il progetto usa percorsi locali per `train.csv` e `test.csv`; per renderlo portabile e' consigliato spostarli in una cartella `data/` e leggere i percorsi da variabili di ambiente.
- Alcuni modelli possono produrre warning di scikit-learn legati a versioni recenti delle API, per esempio sulla deprecazione del parametro `penalty` in `LogisticRegression`.
- Per un uso production-ready servirebbero persistenza dei modelli, test automatici, configurazione esterna e gestione errori piu strutturata.

## Verifica rapida eseguita

Ambiente testato con la `.venv` locale del progetto.

Flusso verificato:

```text
GET  /health                 -> 200
GET  /stats/correlation      -> 200
POST /dataset/preprocess     -> 200
GET  /dataset/get_processed  -> 200
POST /logreg/train           -> 200
GET  /models/list            -> 200
POST /predict/test           -> 200, output CSV
```

## Possibili miglioramenti

- Aggiungere un file `requirements.txt`.
- Aggiungere una cartella `data/` esclusa da Git tramite `.gitignore`.
- Usare `Pipeline` e `ColumnTransformer` di scikit-learn per centralizzare preprocessing e modello.
- Salvare modelli addestrati con `joblib`.
- Aggiungere test automatici con `pytest`.
- Esporre una configurazione tramite `.env` invece di percorsi hardcoded.

## Licenza

Questo progetto e' pensato per scopi didattici e sperimentali. Aggiungi una licenza esplicita prima di pubblicarlo come repository open source.
