import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from it.akron_valtellina.data.preprocessing import CleanDataset


class Visualizzatore:
    """
    Classe per la visualizzazione dei dati.
    """

    def __init__(self, clean_dataset_instance):
        """
        Parameters:
        -----------
        clean_dataset_instance : CleanDataset
            Istanza della classe CleanDataset già caricata
        """
        if not isinstance(clean_dataset_instance, CleanDataset):
            raise ValueError("clean_dataset_instance deve essere un'istanza di CleanDataset")

        self.__clean_dataset = clean_dataset_instance
        self.__X = clean_dataset_instance._CleanDataset__X
        self.__y = clean_dataset_instance._CleanDataset__y

    def grafici_distribuzioni(self):
        """
        Grafici distribuzioni per tutte le variabili.
        Per variabili categoriche: bar plot
        Per variabili numeriche: histogram

        Returns:
        --------
        bytes: Immagine PNG in formato bytes
        """
        df = self.__clean_dataset.replace_strange_missing_values()

        cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
        numeric_cols = df.select_dtypes(include='number').columns

        if len(cat_cols) > 0:
            n_cols = 3
            n_rows = int(np.ceil(len(cat_cols) / n_cols))

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6 * n_cols, 4 * n_rows)
            )

            # Gestisci il caso con una sola riga
            if n_rows == 1:
                axes = np.array([axes])

            axes = axes.flatten()

            for idx, col in enumerate(cat_cols):
                df[col].value_counts(dropna=False).plot(
                    kind='bar',
                    ax=axes[idx]
                )

                axes[idx].set_title(f'Distribuzione: {col}')
                axes[idx].set_xlabel('')
                axes[idx].tick_params(axis='x', rotation=45)

            # Elimina assi inutilizzati
            for idx in range(len(cat_cols), len(axes)):
                fig.delaxes(axes[idx])

            plt.tight_layout()

        else:
            # Solo variabili numeriche
            fig, axes = plt.subplots(figsize=(12, 10))
            df[numeric_cols].hist(bins=20, ax=axes)
            axes.set_title('Distribuzione variabili numeriche')
            plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

    def grafici_boxplot(self):
        """
        Grafici boxplot per tutte le variabili numeriche.

        Returns:
        --------
        bytes: Immagine PNG in formato bytes
        """
        X_num = self.__X.select_dtypes(include="number")

        if X_num.empty:
            raise ValueError("Nessuna colonna numerica trovata per il boxplot")

        fig, ax = plt.subplots(figsize=(12, 8))
        X_num.boxplot(rot=45, ax=ax)
        ax.set_title('Boxplot delle variabili numeriche')

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf

    def matrice_correlazione(self, metodo="pearson"):
        """
        Calcola la matrice di correlazione.

        Parameters:
        -----------
        metodo : str, default='pearson'
            Metodo di correlazione: 'pearson', 'spearman', 'kendall'

        Returns:
        --------
        dict: Matrice di correlazione in formato JSON
        """
        df = self.__clean_dataset.replace_strange_missing_values()
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            return {'error': 'Nessuna colonna numerica per il calcolo della correlazione'}

        corr_matrix = numeric_df.corr(method=metodo)

        # Converti in formato JSON
        result = {
            'method': metodo,
            'columns': corr_matrix.columns.tolist(),
            'correlation_matrix': corr_matrix.values.tolist(),
            'shape': list(corr_matrix.shape)
        }

        return result

    def grafici_correlazione(self, metodo="pearson"):
        """
        Grafico della matrice di correlazione (heatmap).

        Parameters:
        -----------
        metodo : str, default='pearson'
            Metodo di correlazione: 'pearson', 'spearman', 'kendall'

        Returns:
        --------
        bytes: Immagine PNG in formato bytes
        """
        df = self.__clean_dataset.replace_strange_missing_values()
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            raise ValueError("Nessuna colonna numerica per il calcolo della correlazione")

        corr = numeric_df.corr(method=metodo)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.8}
        )

        ax.set_title(f"Matrice di Correlazione ({metodo.title()})")

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        return buf

    def grafico_pairplot(self, vars=None, hue=None, sample_size=1000):
        """
        Pairplot per visualizzare relazioni tra variabili numeriche.

        Parameters:
        -----------
        vars : list, optional
            Lista di colonne da includere
        hue : str, optional
            Colonna per colorare i punti
        sample_size : int, default=1000
            Numero massimo di campioni da usare (per performance)

        Returns:
        --------
        bytes: Immagine PNG in formato bytes
        """
        df = self.__clean_dataset.replace_strange_missing_values()
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            raise ValueError("Nessuna colonna numerica per il pairplot")

        # Campionamento per performance
        if len(numeric_df) > sample_size:
            numeric_df = numeric_df.sample(n=sample_size, random_state=42)

        # Seleziona colonne
        if vars is None:
            vars = numeric_df.columns[:5]  # Limita a 5 colonne per performance

        # Crea pairplot
        if hue and hue in df.columns:
            plot_data = numeric_df.copy()
            plot_data[hue] = df.loc[plot_data.index, hue]
            g = sns.pairplot(plot_data, vars=vars, hue=hue, diag_kind='kde')
        else:
            g = sns.pairplot(numeric_df[vars], diag_kind='kde')

        fig = g.fig
        fig.set_size_inches(15, 12)

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        return buf

    def grafico_target_per_categoria(self, categorical_col=None):
        """
        Grafico della distribuzione del target per categoria.

        Parameters:
        -----------
        categorical_col : str, optional
            Colonna categorica da analizzare (se None, usa tutte)

        Returns:
        --------
        bytes: Immagine PNG in formato bytes
        """
        if self.__y is None:
            raise ValueError("Nessun target disponibile")

        df = self.__clean_dataset.replace_strange_missing_values()

        if categorical_col:
            # Analisi per una singola colonna
            fig, ax = plt.subplots(figsize=(10, 6))

            # Raggruppa per colonna categorica e target
            grouped = df.groupby(
                [categorical_col, self.__y.name if hasattr(self.__y, 'name') else 'target']).size().unstack()
            grouped.plot(kind='bar', ax=ax)
            ax.set_title(f'Distribuzione del target per {categorical_col}')
            ax.set_xlabel(categorical_col)
            ax.set_ylabel('Conteggio')
            ax.legend(title='Target')
            plt.xticks(rotation=45)

        else:
            # Grafico a torta del target
            fig, ax = plt.subplots(figsize=(8, 8))
            target_values = self.__y.value_counts()
            ax.pie(target_values.values, labels=target_values.index.astype(str), autopct='%1.1f%%', startangle=90)
            ax.set_title('Distribuzione del Target')

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        return buf


