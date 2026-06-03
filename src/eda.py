"""
eda.py
------
Funções para análise exploratória de dados (EDA) e visualizações.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_loader import NUMERIC_COLS, TARGET_COL

# Configuração visual padrão
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# ---------------------------------------------------------------------------
# Distribuições
# ---------------------------------------------------------------------------

def plot_quality_distribution(df: pd.DataFrame) -> None:
    """
    Plota a distribuição da nota original de qualidade (`quality`).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna `quality`.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="quality", palette="Blues_d")
    plt.title("Distribuição das notas de qualidade dos vinhos")
    plt.xlabel("Nota de qualidade")
    plt.ylabel("Quantidade de vinhos")
    plt.tight_layout()
    plt.show()


def plot_binary_target_distribution(df: pd.DataFrame) -> None:
    """
    Plota a distribuição da variável alvo binária `high_quality`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo a coluna `high_quality`.
    """
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x=TARGET_COL, palette="Set2")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baixa/Média qualidade", "Alta qualidade"])
    plt.title("Distribuição das classes de qualidade")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade de vinhos")
    plt.tight_layout()
    plt.show()

    dist = df[TARGET_COL].value_counts(normalize=True) * 100
    print(f"Alta qualidade   (1): {dist.get(1, 0):.1f}%")
    print(f"Baixa/Média qual (0): {dist.get(0, 0):.1f}%")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """
    Plota histogramas de todas as variáveis físico-químicas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo as colunas numéricas.
    """
    df[NUMERIC_COLS].hist(figsize=(18, 14), bins=30, color="#4878CF", edgecolor="white")
    plt.suptitle("Distribuição das variáveis físico-químicas", fontsize=16)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Comparações por classe
# ---------------------------------------------------------------------------

def plot_boxplots_by_class(df: pd.DataFrame) -> None:
    """
    Plota boxplots de cada variável numérica separados pela classe de qualidade.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo as variáveis numéricas e a coluna `high_quality`.
    """
    for col in NUMERIC_COLS:
        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(data=df, x=TARGET_COL, y=col, palette="Set2")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baixa/Média", "Alta"])
        plt.title(f"{col} por classe de qualidade")
        plt.xlabel("Classe")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Correlação
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plota a matriz de correlação completa do DataFrame (excluindo a coluna `Id`).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com as variáveis numéricas.
    """
    df_corr = df.drop(columns=["Id"], errors="ignore")
    corr_matrix = df_corr.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
    )
    plt.title("Matriz de correlação")
    plt.tight_layout()
    plt.show()


def plot_correlation_with_target(df: pd.DataFrame) -> None:
    """
    Plota a correlação de cada variável com a variável alvo `high_quality`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo as variáveis e a coluna `high_quality`.
    """
    df_corr = df.drop(columns=["Id"], errors="ignore")
    corr_target = df_corr.corr()[TARGET_COL].drop(TARGET_COL).sort_values()

    corr_target.plot(kind="barh", figsize=(10, 7), color="#4878CF")
    plt.title("Correlação das variáveis com alta qualidade")
    plt.xlabel("Correlação")
    plt.ylabel("Variáveis")
    plt.tight_layout()
    plt.show()

    return corr_target


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def plot_outlier_boxplots(df: pd.DataFrame) -> None:
    """
    Plota boxplots individuais de cada variável para visualização de outliers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo as variáveis numéricas.
    """
    for col in NUMERIC_COLS:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col], color="#4878CF")
        plt.title(f"Boxplot - {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()


def compute_outliers_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o resumo de outliers pelo método IQR para cada variável numérica.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo as variáveis numéricas.

    Returns
    -------
    pd.DataFrame
        Tabela com limites e quantidade de outliers por variável,
        ordenada de forma decrescente pela quantidade.
    """
    summary = []

    for col in NUMERIC_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()

        summary.append({
            "variavel": col,
            "limite_inferior": round(lower, 4),
            "limite_superior": round(upper, 4),
            "qtd_outliers": n_out,
            "percentual_outliers": round(n_out / len(df) * 100, 2),
        })

    return pd.DataFrame(summary).sort_values("qtd_outliers", ascending=False).reset_index(drop=True)
