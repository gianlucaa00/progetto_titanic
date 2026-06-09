import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class Split:
    def __init__(self, X, y, dummies=False, test_size = 0.2, random_state = 42):
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

        # Encoding del target
        if isinstance(self.__y, pd.DataFrame):
            y = self.__y.squeeze()
        else:
            y = self.__y

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        return X_dumm, y_encoded

    def get_split(self, dummies=False, test_size=0.2, random_state=42):

        if not dummies:
            X_train, X_test, y_train, y_test = train_test_split(
                self.__X,
                self.__y,
                test_size=test_size,
                random_state=random_state
            )
        else:
            X_dumm, y_encoded = self.get_dummies()
            X_train, X_test, y_train, y_test = train_test_split(
                X_dumm,
                y_encoded,
                test_size=test_size,
                random_state=random_state,
                stratify=y_encoded
            )

        return X_train, X_test, y_train, y_test
