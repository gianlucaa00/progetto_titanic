from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score
)
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns


class Evaluation:
    def __init__(self, y_test, y_pred, classification=False):
        self.__y_test = y_test
        self.__y_pred = y_pred
        self.__classification = classification
        self.__results = None

    def metrics(self, y_test=None, y_pred=None, classification=None):
        """
        Calcola le metriche di valutazione.

        Parameters:
        -----------
        y_test : array-like, optional
            Valori reali (se non forniti, usa quelli dell'istanza)
        y_pred : array-like, optional
            Valori predetti (se non forniti, usa quelli dell'istanza)
        classification : bool, optional
            Se True, calcola metriche di classificazione.
            Se False, calcola metriche di regressione.
        """
        # Usa i parametri passati o quelli dell'istanza
        y_test = y_test if y_test is not None else self.__y_test
        y_pred = y_pred if y_pred is not None else self.__y_pred
        classification = classification if classification is not None else self.__classification

        if not classification:
            # Metriche di regressione
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)


            self.__results = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
        else:
            # Metriche di classificazione
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)


            self.__results = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Balanced Accuracy': balanced_acc,
                'Confusion Matrix': conf_matrix,
                'y_test': y_test,
                'y_pred': y_pred
            }

        return self.__results

    def grafico_matrice_confusione(self, classification=None):
        """
        Genera il grafico della matrice di confusione.
        """
        # CORREZIONE 4: Verifica che sia classificazione
        classification = classification if classification is not None else self.__classification

        if not classification:
            raise ValueError("La matrice di confusione è disponibile solo per problemi di classificazione")

        # CORREZIONE 5: Calcola metriche se non sono già state calcolate
        if self.__results is None:
            self.metrics(classification=classification)

        # CORREZIONE 6: Verifica che la Confusion Matrix esista nei risultati
        if 'Confusion Matrix' not in self.__results:
            raise ValueError("Devi prima calcolare le metriche con metrics(classification=True)")

        cm = self.__results['Confusion Matrix']

        # Usa y_test e y_pred salvati nei risultati o quelli dell'istanza
        if 'y_test' in self.__results and 'y_pred' in self.__results:
            y_test = self.__results['y_test']
            y_pred = self.__results['y_pred']
        else:
            y_test = self.__y_test
            y_pred = self.__y_pred

        classes = np.unique(np.concatenate([y_test, y_pred]))

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            ax=ax
        )
        ax.set_xlabel('Predetto')
        ax.set_ylabel('Reale')
        ax.set_title('Matrice di Confusione')
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    def get_results(self):
        """Restituisce i risultati delle metriche."""
        if self.__results is None:
            raise ValueError("Devi prima calcolare le metriche con metrics()")
        return self.__results