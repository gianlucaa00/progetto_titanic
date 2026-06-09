## Spiegazione delle chiamate API

Ecco una spiegazione dettagliata di ogni endpoint e cosa fa:

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
# 1. Preprocessa e salva i dati
curl -X POST /dataset/preprocess -d '{"save_state": true}'

# 2. Verifica stato
curl GET /dataset/status

# 3. Addestra un modello
curl -X POST /logreg/train -d '{"C": 1.0}'

# 4. Ottieni riepilogo
curl GET /logreg/summary

# 5. Esegui cross-validation
curl -X POST /logreg/cv -d '{"k": 5}'

# 6. Ottimizza iperparametri
curl -X POST /finetuning/grid -d '{
    "model_type": "logistic",
    "param_grid": {"C": [0.1, 1.0, 10.0]},
    "cv": 5
}'
```