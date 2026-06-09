import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score


class RandomForest:
    def __init__(self, X_train, X_val, y_train, y_val):
        self.__X_train = X_train
        self.__X_val = X_val
        self.__y_train = y_train
        self.__y_val = y_val

        self.__best_model = None
        self.__best_params = None
        self.__cv_best_score = None
        self.__accuracy_val = None

    def ottimizza_iperparametri(self, cv=2):
        base_model = RandomForestClassifier(
            random_state=42
        )

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"]
        }

        kfold = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=42
        )

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=kfold,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )

        grid.fit(self.__X_train, self.__y_train)

        self.__best_model = grid.best_estimator_
        self.__best_params = grid.best_params_
        self.__cv_best_score = grid.best_score_

        return self.__best_params, self.__cv_best_score

    def addestra_modello(self):
        if self.__best_model is None:
            self.__best_model = RandomForestClassifier(
                random_state=42
            )

        self.__best_model.fit(self.__X_train, self.__y_train)

    def accuracy_validation(self):
        y_pred = self.__best_model.predict(self.__X_val)
        self.__accuracy_val = accuracy_score(self.__y_val, y_pred)
        return self.__accuracy_val

    def mostra_risultati(self):
        return {
            "best_params": self.__best_params,
            "accuracy_media_cv": self.__cv_best_score,
            "accuracy_validation": self.__accuracy_val
        }
if __name__ == "__main__":
    import pandas as pd
    from pacchetti.modelli.split_validation import Split

    df = pd.read_csv(r"C:\Users\HP\Downloads\train.csv")

    X = df.drop(columns=["Survived"], errors="ignore")
    y = df["Survived"]

    splitter = Split(X, y)
    X_train, X_val, y_train, y_val = splitter.get_split()

    modello = RandomForest(X_train, X_val, y_train, y_val)

    best_params, best_cv = modello.ottimizza_iperparametri()
    print("Migliori iperparametri:", best_params)
    print(f"Accuracy media CV migliore: {best_cv:.4f}")

    modello.addestra_modello()

    acc_val = modello.accuracy_validation()
    print(f"Accuracy validation: {acc_val:.4f}")

    print(modello.mostra_risultati())