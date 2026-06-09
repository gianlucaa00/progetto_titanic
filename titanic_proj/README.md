## Spiegazione delle chiamate API

## 🗂️ **ENDPOINT DATASET**

### **`GET /dataset/info`** - Informazioni generali del dataset

**Cosa fa:**
- Recupera informazioni di base sul dataset caricato
- Non modifica i dati, solo lettura

**Risposta:**
```json
{
    "source": "csv",
    "shape": [891, 12],
    "columns": ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", ...],
    "numeric_cols": ["PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"],
    "categorical_cols": ["Name", "Sex", "Ticket", "Cabin", "Embarked"],
    "has_target": true,
    "target_shape": [891],
    "target_unique": 2
}
```

---

### **`GET /dataset/na`** - Analisi valori mancanti

**Cosa fa:**
- Identifica e quantifica i valori mancanti nel dataset
- Calcola percentuali di missing per ogni colonna

**Risposta:**
```json
{
    "null_values": {"Age": 177, "Cabin": 687, "Embarked": 2},
    "missing_percent": {"Age": 19.87, "Cabin": 77.10, "Embarked": 0.22},
    "total_missing": 866,
    "columns_with_missing": 3
}
```

**Utilità:** Capire quali colonne hanno molti missing per decidere se eliminarle o imputarle.

---

### **`POST /dataset/preprocess`** - Preprocessing del dataset

**Cosa fa:**
- Applica tutte le trasformazioni di pulizia dei dati
- **Salva lo stato** se `save_state: true` per usarlo nei modelli successivi

**Parametri (body JSON):**
| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `remove_columns` | `[]` | Lista di colonne da eliminare |
| `imputation_method` | `"statistical"` | `"statistical"` (media/modana) o `"knn"` |
| `knn_neighbors` | `5` | Numero di vicini per KNN |
| `remove_outliers` | `false` | Se rimuovere gli outliers |
| `outlier_method` | `"iqr"` | `"iqr"` o `"zscore"` |
| `outlier_threshold` | `1.5` | Soglia per outlier detection |
| `save_state` | `false` | Salva i dati processati per i modelli |

**Flusso delle operazioni:**
1. 🔹 Elimina colonne specificate dall'utente
2. 🔹 Rimuove colonne con >50% di valori mancanti
3. 🔹 Elimina righe duplicate
4. 🔹 Imputa valori mancanti (statistico o KNN)
5. 🔹 Rimuove colonne costanti (un solo valore)
6. 🔹 (Opzionale) Rimuove outliers
7. 🔹 Salva lo stato se richiesto

**Risposta:**
```json
{
    "preprocessing_info": {
        "columns_removed": ["Cabin", "Ticket", "Name"],
        "imputation_method": "knn",
        "final_shape": [891, 8],
        "state_saved": true
    },
    "data_shape": [891, 8],
    "target_info": {
        "unique_values": 2,
        "distribution": {"0": 549, "1": 342},
        "class_balance": {"0": 0.616, "1": 0.384}
    }
}
```

---

### **`GET /dataset/get_processed`** - Recupera dati preprocessati

**Cosa fa:**
- Restituisce le informazioni dei dati salvati nella cache
- Utile per verificare lo stato del preprocessing

**Risposta:**
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

**Cosa fa:**
- Cancella i dati preprocessati salvati in memoria
- Forza i prossimi modelli a riprocessare i dati

**Risposta:**
```json
{
    "message": "Stato preprocessato resettato con successo"
}
```

---

### **`DELETE /dataset/columns`** - Elimina colonne

**Cosa fa:**
- Rimuove colonne specificate dal dataset
- Operazione immediata senza preprocessing aggiuntivo

**Body:**
```json
{
    "columns": ["PassengerId", "Ticket", "Cabin"]
}
```

**Risposta:**
```json
{
    "columns_dropped": ["PassengerId", "Ticket", "Cabin"],
    "remaining_columns": ["Pclass", "Name", "Sex", "Age", ...],
    "new_shape": [891, 9]
}
```

---

## 📈 **ENDPOINT STATISTICHE**

### **`GET /stats/descriptive`** - Statistiche descrittive complete

**Cosa fa:**
- Calcola statistiche dettagliate per tutte le colonne
- Include media, mediana, deviazione standard, quartili, skewness, kurtosis

**Risposta:**
```json
{
    "dataset_info": {
        "name": "Original Dataset",
        "shape": [891, 12],
        "total_cells": 10692
    },
    "missing_values": {
        "total_missing": 866,
        "total_missing_percent": 8.1
    },
    "numeric_columns_stats": {
        "Age": {
            "mean": 29.7,
            "median": 28.0,
            "std": 14.5,
            "min": 0.42,
            "max": 80.0,
            "skewness": 0.39,
            "missing_percent": 19.87
        }
    },
    "categorical_columns_stats": {
        "Sex": {
            "unique_values": 2,
            "most_frequent": "male",
            "frequencies": {"male": 577, "female": 314}
        }
    }
}
```

---

### **`GET /stats/quick`** - Riepilogo rapido

**Cosa fa:**
- Statistiche sintetiche per una visione d'insieme veloce

**Risposta:**
```json
{
    "rows": 891,
    "columns": 12,
    "numeric_columns": 7,
    "categorical_columns": 5,
    "missing_values_total": 866,
    "missing_percent": 8.1,
    "duplicates": 0,
    "memory_usage_mb": 0.15
}
```

---

### **`GET /stats/correlation`** - Matrice di correlazione

**Parametri (query string):**
- `method`: `pearson` (default), `spearman`, `kendall`

**Cosa fa:**
- Calcola le correlazioni tra tutte le variabili numeriche

**Risposta:**
```json
{
    "method": "pearson",
    "columns": ["Pclass", "Age", "SibSp", "Parch", "Fare"],
    "correlation_matrix": [
        [1.0, -0.34, 0.08, 0.02, -0.55],
        [-0.34, 1.0, -0.12, -0.04, 0.10],
        [0.08, -0.12, 1.0, 0.41, 0.16],
        [0.02, -0.04, 0.41, 1.0, 0.22],
        [-0.55, 0.10, 0.16, 0.22, 1.0]
    ]
}
```

---

## 🎨 **ENDPOINT VISUALIZZAZIONI**

Tutti gli endpoint di visualizzazione restituiscono **immagini PNG** (non JSON).

### **`GET /plots/distributions`** - Istogrammi e bar plot

**Cosa fa:**
- Per variabili numeriche: istogrammi della distribuzione
- Per variabili categoriche: bar plot delle frequenze

**Output:** Immagine PNG con griglia di grafici

---

### **`GET /plots/boxplot`** - Boxplot

**Cosa fa:**
- Crea boxplot per tutte le variabili numeriche
- Mostra mediane, quartili e outliers

**Output:** Immagine PNG con boxplot affiancati

**Utilità:** Identificare visivamente gli outliers e la distribuzione dei dati.

---

### **`GET /plots/correlation`** - Heatmap di correlazione

**Parametri (query string):**
- `method`: `pearson` (default), `spearman`, `kendall`

**Cosa fa:**
- Crea una heatmap colorata della matrice di correlazione
- Colori: rosso (correlazione positiva), blu (correlazione negativa)

**Output:** Immagine PNG della heatmap

---

### **`GET /plots/pairplot`** - Pairplot

**Parametri (query string):**
- `vars`: Lista colonne separate da virgola (es. `Age,Fare,Pclass`)
- `hue`: Colonna per colorare i punti (es. `Sex`, `Survived`)

**Cosa fa:**
- Crea una matrice di scatter plot tra variabili numeriche
- Sulle diagonali: distribuzione di ogni variabile
- Colori differenti per categoria (se specificato `hue`)

**Output:** Immagine PNG del pairplot

**Esempio:** 
```
/plots/pairplot?vars=Age,Fare,Pclass&hue=Survived
```


---

### 📊 **LOGISTIC REGRESSION (Regressione Logistica)**

#### **`POST /logreg/train`** - Addestra il modello
```json
// Body richiesta (opzionale)
{
    "use_pca": false,      // Se true, applica PCA prima del training
    "n_components": null,  // Numero componenti PCA (se null, automatico)
    "C": 1.0,             // Inverse della regolarizzazione (più piccolo = più regolarizzazione)
    "penalty": "l2",      // Tipo di regolarizzazione: 'l1', 'l2', 'elasticnet'
    "max_iter": 1000      // Numero massimo di iterazioni
}
```
**Cosa fa:**
1. Recupera i dati preprocessati dalla cache
2. Crea un'istanza di `RegressioneLogistica`
3. Addestra il modello con i parametri specificati
4. Restituisce l'accuratezza su train e test

**Risposta:**
```json
{
    "accuracy_train": 0.823,
    "accuracy_test": 0.802,
    "params": {"C": 1.0, "penalty": "l2", "max_iter": 1000},
    "use_pca": false
}
```

---

#### **`GET /logreg/summary`** - Riassunto del modello
**Cosa fa:**
1. Recupera i dati dalla cache
2. Addestra il modello
3. Restituisce un riepilogo completo con coefficienti e intercetta

**Risposta:**
```json
{
    "model_type": "Logistic Regression",
    "accuracy_train": 0.823,
    "accuracy_test": 0.802,
    "coefficients": [
        {"variabile": "Pclass", "coefficiente": -1.2, "odds_ratio": 0.30},
        {"variabile": "Age", "coefficiente": -0.05, "odds_ratio": 0.95}
    ],
    "intercept": 0.5
}
```

---

#### **`POST /logreg/cv`** - Cross-Validation
```json
// Body richiesta
{
    "k": 5,              // Numero di fold
    "method": "kfold"    // 'kfold', 'loo', 'lho'
}
```
**Cosa fa:**
1. Esegue K-Fold Cross-Validation sul modello
2. Calcola accuratezza per ogni fold
3. Restituisce statistiche aggregate

**Risposta:**
```json
{
    "mean_score": 0.795,
    "std_score": 0.023,
    "scores_per_fold": [0.78, 0.81, 0.79, 0.80, 0.79],
    "min_score": 0.78,
    "max_score": 0.81,
    "n_folds": 5,
    "method": "kfold",
    "k": 5
}
```

---

### 🔷 **SVM (Support Vector Machine)**

#### **`POST /svc/train`**
```json
{
    "use_pca": false,
    "n_components": null,
    "kernel": "rbf",     // 'linear', 'poly', 'rbf', 'sigmoid'
    "C": 1.0,           // Parametro di regolarizzazione
    "gamma": "scale"    // 'scale', 'auto' o valore numerico
}
```
**Cosa fa:** Addestra SVM con i parametri specificati

---

#### **`GET /svc/summary`**
Restituisce riepilogo del modello SVM con kernel e parametri usati.

---

#### **`POST /svc/cv`**
Esegue cross-validation per SVM.

---

### 🌲 **RANDOM FOREST**

#### **`POST /rf/train`**
```json
{
    "use_pca": false,
    "n_components": null,
    "n_estimators": 100,    // Numero di alberi
    "max_depth": null,      // Profondità massima (null = illimitata)
    "min_samples_split": 2  // Minimo campioni per split
}
```

---

#### **`GET /rf/summary`**
Restituisce:
- Accuratezza su train/test
- **Feature importance** (top 10 variabili più importanti)

```json
{
    "model_type": "Random Forest",
    "accuracy_train": 0.95,
    "accuracy_test": 0.81,
    "feature_importance": [
        {"variabile": "Sex", "importanza": 0.35},
        {"variabile": "Pclass", "importanza": 0.25}
    ],
    "params": {"n_estimators": 100, "max_depth": null}
}
```

---

#### **`POST /rf/cv`**
Esegue cross-validation per Random Forest.

---

### ⚡ **XGBOOST**

#### **`POST /xgb/train`**
```json
{
    "use_pca": false,
    "n_components": null,
    "n_estimators": 100,    // Numero di alberi
    "max_depth": 3,         // Profondità massima
    "learning_rate": 0.1    // Tasso di apprendimento
}
```

---

#### **`GET /xgb/summary`**
Restituisce accuratezza e feature importance.

---

#### **`POST /xgb/cv`**
Esegue cross-validation per XGBoost.

---

### 🎯 **FINE TUNING (Ottimizzazione Iperparametri)**

#### **`POST /finetuning/grid`** - Grid Search
```json
{
    "model_type": "logistic",  // 'logistic', 'svm', 'randomforest', 'xgboost'
    "param_grid": {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"]
    },
    "cv": 5
}
```
**Cosa fa:**
1. Testa TUTTE le combinazioni di parametri specificate
2. Per ogni combinazione, esegue cross-validation
3. Restituisce la combinazione migliore

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

#### **`POST /finetuning/random`** - Random Search
```json
{
    "model_type": "randomforest",
    "param_distributions": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, null]
    },
    "cv": 5,
    "n_iter": 10     // Numero di combinazioni casuali da testare
}
```
**Cosa fa:**
1. Testa un numero limitato di combinazioni casuali (n_iter)
2. Più veloce di Grid Search per spazi di parametri grandi
3. Restituisce la migliore combinazione trovata

---

## 📋 **RIASSUNTO DEI 4 MODELLI**

| Modello | Train (POST) | Summary (GET) | CV (POST) |
|---------|--------------|---------------|-----------|
| **Logistic Regression** | `/logreg/train` | `/logreg/summary` | `/logreg/cv` |
| **SVM** | `/svc/train` | `/svc/summary` | `/svc/cv` |
| **Random Forest** | `/rf/train` | `/rf/summary` | `/rf/cv` |
| **XGBoost** | `/xgb/train` | `/xgb/summary` | `/xgb/cv` |

## 🔄 **Flusso di lavoro tipico**


```bash
# 1. Esplora il dataset
GET /dataset/info
GET /dataset/na
GET /stats/quick

# 2. Visualizza i dati
GET /plots/distributions
GET /plots/boxplot
GET /plots/correlation

# 3. Preprocessa e salva
POST /dataset/preprocess
{
    "remove_columns": ["PassengerId", "Name", "Ticket", "Cabin"],
    "imputation_method": "knn",
    "knn_neighbors": 5,
    "remove_outliers": true,
    "save_state": true
}

# 4. Verifica stato
GET /dataset/get_processed

# 5. Addestra modelli (usano i dati preprocessati)
POST /logreg/train

# 6. Se necessario, reset della cache
POST /dataset/reset
```
