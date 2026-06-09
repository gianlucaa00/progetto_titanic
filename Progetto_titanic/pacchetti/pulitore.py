import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class Pulitore:
    def __init__(self):
        self.__train = pd.read_csv(r"C:\Users\HP\Downloads\train.csv")
        self.__best_params = None

    def rimuovi_colonne(self):
        self.__train = self.__train.drop(
            columns=["PassengerId", "Cabin", "Embarked", "Ticket", "Name"],
            errors="ignore"
        )

    def imputa_na(self):
        X = self.__train.drop(columns=["Survived"], errors="ignore")
        y = self.__train["Survived"]

        X = pd.get_dummies(X, drop_first=True)

        pipeline = Pipeline([
            ("imputer", KNNImputer()),
            ("model", RandomForestClassifier(random_state=42))
        ])

        param_grid = {
            "imputer__n_neighbors": [3, 5, 7, 9, 11],
            "imputer__weights": ["uniform", "distance"],
            "imputer__metric": ["nan_euclidean"],
            "imputer__add_indicator": [False]
        }

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid.fit(X, y)

        self.__best_params = grid.best_params_

        imputer = KNNImputer(
            n_neighbors=self.__best_params["imputer__n_neighbors"],
            weights=self.__best_params["imputer__weights"],
            metric=self.__best_params["imputer__metric"],
            add_indicator=self.__best_params["imputer__add_indicator"]
        )

        X_imputato = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        self.__train = pd.concat([X_imputato, y], axis=1)

        return self.__train

    def mostra_parametri_migliori(self):
        return self.__best_params

    def grafici_distribuzione(self):

        colonne_numeriche = self.__train.select_dtypes(include=["int64", "float64"]).columns

        for colonna in colonne_numeriche:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.__train[colonna], kde=False)
            plt.title(f"Distribuzione di {colonna}")
            plt.xlabel(colonna)
            plt.ylabel("Frequenza")
            plt.show()
pulitore = Pulitore()
pulitore.rimuovi_colonne()

train_pulito = pulitore.imputa_na()

print(train_pulito.head())
print(pulitore.mostra_parametri_migliori())
pulitore.grafici_distribuzione()