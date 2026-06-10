## 📚 **DOCUMENTAZIONE COMPLETA DELLE API**

---

## 🗂️ **ENDPOINT DATASET**

### **`GET /dataset/info`** - Informazioni generali del dataset

**Cosa fa:** Restituisce informazioni di base sul dataset caricato (shape, colonne, tipi di dati).

**Risposta esempio:**
```json
{
    "source": "csv",
    "shape": [891, 12],
    "columns": ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
    "numeric_cols": ["PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"],
    "categorical_cols": ["Name", "Sex", "Ticket", "Cabin", "Embarked"],
    "has_target": true,
    "target_shape": [891],
    "target_unique": 2
}
```

---

### **`GET /dataset/na`** - Analisi valori mancanti

**Cosa fa:** Identifica e quantifica i valori mancanti in ogni colonna.

**Risposta esempio:**
```json
{
    "null_values": {"Age": 177, "Cabin": 687, "Embarked": 2},
    "missing_percent": {"Age": 19.87, "Cabin": 77.10, "Embarked": 0.22},
    "total_missing": 866,
    "columns_with_missing": 3
}
```

---

### **`POST /dataset/preprocess`** - Preprocessing del dataset

**Cosa fa:** Pulisce e prepara i dati per il training. Salva lo stato se `save_state: true`.

**Body JSON (opzionale):**
```json
{
    "remove_columns": ["PassengerId", "Name", "Ticket", "Cabin"],
    "imputation_method": "statistical",
    "knn_neighbors": 5,
    "remove_outliers": false,
    "outlier_method": "iqr",
    "outlier_threshold": 1.5,
    "save_state": true
}
```

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `remove_columns` | `[]` | Lista di colonne da eliminare |
| `imputation_method` | `"statistical"` | `"statistical"` (media/modana) o `"knn"` |
| `knn_neighbors` | `5` | Numero di vicini per KNN |
| `remove_outliers` | `false` | Se rimuovere gli outliers |
| `outlier_method` | `"iqr"` | `"iqr"` o `"zscore"` |
| `outlier_threshold` | `1.5` | Soglia per outlier detection |
| `save_state` | `false` | Salva i dati processati per i modelli |

**Risposta esempio:**
```json
{
    "preprocessing_info": {
        "columns_removed": ["PassengerId", "Name", "Ticket", "Cabin"],
        "imputation_method": "statistical",
        "final_shape": [891, 8],
        "state_saved": true
    },
    "data_shape": [891, 8],
    "target_info": {
        "unique_values": 2,
        "distribution": {"0": 549, "1": 342}
    }
}
```

---

### **`GET /dataset/get_processed`** - Recupera dati preprocessati

**Cosa fa:** Restituisce informazioni sui dati salvati in cache dopo il preprocessing.

**Risposta esempio:**
```json
{
    "is_processed": true,
    "data_shape": [891, 8],
    "columns": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
    "target_distribution": {"0": 549, "1": 342}
}
```

---

### **`POST /dataset/reset`** - Reset della cache

**Cosa fa:** Cancella i dati preprocessati salvati in memoria.

**Risposta:**
```json
{
    "message": "Stato preprocessato resettato con successo"
}
```

---

## 📊 **ENDPOINT STATISTICHE**

### **`GET /stats/descriptive`** - Statistiche descrittive complete

**Cosa fa:** Calcola statistiche dettagliate per tutte le colonne (media, mediana, std, quartili, skewness, kurtosis).

**Risposta esempio:**
```json
{
    "dataset_info": {
        "name": "Original Dataset",
        "shape": [891, 12]
    },
    "missing_values": {
        "total_missing": 866,
        "missing_percent": 8.1
    },
    "numeric_columns_stats": {
        "Age": {
            "mean": 29.7,
            "median": 28.0,
            "std": 14.5,
            "missing_percent": 19.87
        }
    }
}
```

---

### **`GET /stats/quick`** - Riepilogo rapido

**Cosa fa:** Restituisce statistiche sintetiche per una visione d'insieme veloce.

**Risposta esempio:**
```json
{
    "rows": 891,
    "columns": 12,
    "numeric_columns": 7,
    "categorical_columns": 5,
    "missing_values_total": 866,
    "missing_percent": 8.1,
    "duplicates": 0
}
```

---

### **`GET /stats/correlation`** - Matrice di correlazione

**Parametri query:** `method` (pearson, spearman, kendall)

**Cosa fa:** Calcola le correlazioni tra tutte le variabili numeriche.

---

## 🎨 **ENDPOINT VISUALIZZAZIONI**

Tutti gli endpoint di visualizzazione restituiscono **immagini PNG**.

| Endpoint | Descrizione |
|----------|-------------|
| `GET /plots/distributions` | Istogrammi per variabili numeriche, bar plot per categoriche |
| `GET /plots/boxplot` | Boxplot per tutte le variabili numeriche |
| `GET /plots/correlation` | Heatmap della matrice di correlazione |
| `GET /plots/pairplot?vars=Age,Fare&hue=Survived` | Pairplot tra variabili numeriche |

---

## 🤖 **ENDPOINT MODELLI**

### **Addestramento modelli (POST)**

| Endpoint | Modello | Parametri principali |
|----------|---------|---------------------|
| `/logreg/train` | Logistic Regression | `C`, `penalty`, `max_iter`, `use_pca` |
| `/svc/train` | SVM | `kernel`, `C`, `gamma`, `use_pca` |
| `/rf/train` | Random Forest | `n_estimators`, `max_depth`, `use_pca` |
| `/xgb/train` | XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `use_pca` |

**Esempio di chiamata:**
```bash
curl -X POST /logreg/train \
  -H "Content-Type: application/json" \
  -d '{"C": 1.0, "penalty": "l2"}'
```

**Risposta esempio:**
```json
{
    "accuracy_train": 0.823,
    "accuracy_test": 0.802,
    "precision_train": 0.82,
    "precision_test": 0.80,
    "recall_train": 0.81,
    "recall_test": 0.79,
    "f1_train": 0.815,
    "f1_test": 0.795,
    "params": {"C": 1.0, "penalty": "l2"}
}
```

---

### **Cross-Validation (POST)**

| Endpoint | Modello |
|----------|---------|
| `/logreg/cv` | Logistic Regression |
| `/svc/cv` | SVM |
| `/rf/cv` | Random Forest |
| `/xgb/cv` | XGBoost |

**Body JSON:**
```json
{
    "k": 5,
    "method": "kfold"
}
```

**Risposta esempio:**
```json
{
    "mean_score": 0.795,
    "std_score": 0.023,
    "scores_per_fold": [0.78, 0.81, 0.79, 0.80, 0.79],
    "min_score": 0.78,
    "max_score": 0.81,
    "n_folds": 5
}
```

---

## 💾 **GESTIONE MODELLI SALVATI**

### **`GET /models/list`** - Lista modelli salvati

**Risposta esempio:**
```json
{
    "models": {
        "logistic": {
            "is_trained": true,
            "train_accuracy": 0.823,
            "test_accuracy": 0.802,
            "model_params": {"C": 1.0}
        }
    },
    "n_models": 1,
    "has_models": true
}
```

---

### **`GET /models/get/<model_type>`** - Info modello specifico

**Esempio:** `/models/get/logistic`

**Risposta:** Metriche complete e parametri del modello richiesto.

---

### **`POST /models/select/<model_type>`** - Seleziona modello per predizioni

**Cosa fa:** Sceglie quale modello usare per le predizioni sul test set.

**Esempio:** `/models/select/randomforest`

**Risposta:**
```json
{
    "message": "Modello randomforest selezionato per le predizioni",
    "model_type": "randomforest",
    "train_accuracy": 0.95,
    "test_accuracy": 0.81
}
```

---

### **`POST /models/clear`** - Svuota cache modelli

**Cosa fa:** Cancella tutti i modelli addestrati dalla memoria.

---

## 🔮 **PREDIZIONI**

### **`POST /predict/test`** - Predizioni su file di test

**Cosa fa:** Carica il file di test da `TEST_CONFIG`, applica lo stesso preprocessing del training, e genera predizioni usando il modello selezionato.

**Body JSON (opzionale):**
```json
{
    "model_type": "randomforest",
    "include_probabilities": true
}
```

**Output:** File CSV con colonne originali + colonna `prediction` + probabilità.

**Esempio di utilizzo:**
```bash
curl -X POST /predict/test \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logistic"}' \
  --output predictions.csv
```

---

## ⚙️ **FINE TUNING**

### **`POST /finetuning/grid`** - Grid Search

**Body JSON:**
```json
{
    "model_type": "logistic",
    "param_grid": {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"]
    },
    "cv": 5
}
```

**Risposta:**
```json
{
    "best_params": {"C": 1.0, "penalty": "l2"},
    "best_score": 0.802,
    "scoring_metric": "accuracy",
    "n_candidates": 6
}
```

---

### **`POST /finetuning/random`** - Random Search

**Body JSON:**
```json
{
    "model_type": "randomforest",
    "param_distributions": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7]
    },
    "cv": 5,
    "n_iter": 10
}
```

---

## 🔍 **HEALTH CHECK**

### **`GET /health`** - Stato del servizio

**Risposta:**
```json
{
    "status": "ok",
    "service": "ML API",
    "version": "1.0.0"
}
```

---

## 📋 **FLUSSO DI LAVORO TIPICO**

```bash
# 1. Preprocessa e salva i dati
curl -X POST /dataset/preprocess -d '{"save_state": true, "remove_columns": ["PassengerId", "Name"]}'

# 2. Addestra un modello
curl -X POST /logreg/train -d '{"C": 1.0}'

# 3. Seleziona il modello per le predizioni
curl -X POST /models/select/logistic

# 4. Fai predizioni sul test set
curl -X POST /predict/test --output predictions.csv

# 5. Oppure specifica il modello direttamente
curl -X POST /predict/test -d '{"model_type": "randomforest"}' --output rf_predictions.csv
```