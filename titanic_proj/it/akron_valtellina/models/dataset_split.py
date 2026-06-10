import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Split:
    def __init__(self, X, y, dummies=False, test_size=0.2, random_state=42):
        self.__X = X
        self.__y = y
        self.__dummies = dummies
        self.__test_size = test_size
        self.__random_state = random_state

    def get_dummies(self):
        # One-Hot Encoding delle feature categoriche
        X_dumm = pd.get_dummies(
            self.__X,
            drop_first=True,
            dtype=int
        )

        # Encoding del target (solo se y non è None)
        if self.__y is not None:
            if isinstance(self.__y, pd.DataFrame):
                y = self.__y.squeeze()
            else:
                y = self.__y

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = None

        return X_dumm, y_encoded

    def get_split(self, dummies=False, test_size=0.2, random_state=42):
        # Usa i parametri passati o quelli di default dell'istanza
        use_dummies = dummies if dummies is not None else self.__dummies
        test_size = test_size if test_size is not None else self.__test_size
        random_state = random_state if random_state is not None else self.__random_state

        if not use_dummies:
            if self.__y is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    self.__X,
                    self.__y,
                    test_size=test_size,
                    random_state=random_state
                )
            else:
                # quando y è None, train_test_split restituisce solo 2 valori
                X_train, X_test = train_test_split(
                    self.__X,
                    test_size=test_size,
                    random_state=random_state
                )
                y_train, y_test = None, None
        else:
            X_dumm, y_encoded = self.get_dummies()
            if y_encoded is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_dumm,
                    y_encoded,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y_encoded
                )
            else:
                # quando y_encoded è None, train_test_split restituisce solo 2 valori
                X_train, X_test = train_test_split(
                    X_dumm,
                    test_size=test_size,
                    random_state=random_state
                )
                y_train, y_test = None, None

        return X_train, X_test, y_train, y_test