"""
data_loader.py
--------------
Funções para carregamento, validação e pré-processamento inicial da base de dados.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

TARGET_COL = "high_quality"
DROP_COLS = ["quality", "Id"]


# ---------------------------------------------------------------------------
# Funções
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega a base de dados a partir de um arquivo CSV.

    Parameters
    ----------
    filepath : str
        Caminho para o arquivo CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame com os dados carregados.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não for encontrado no caminho informado.
    ValueError
        Se o arquivo estiver vazio ou não contiver as colunas esperadas.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    if df.empty:
        raise ValueError("O arquivo CSV está vazio.")

    missing_cols = [col for col in NUMERIC_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas ausentes na base: {missing_cols}")

    print(f"[data_loader] Base carregada com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas.")
    return df


def create_binary_target(df: pd.DataFrame, quality_threshold: int = 7) -> pd.DataFrame:
    """
    Cria a variável alvo binária `high_quality` a partir da coluna `quality`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original contendo a coluna `quality`.
    quality_threshold : int, optional
        Nota mínima para classificar um vinho como alta qualidade. Default: 7.

    Returns
    -------
    pd.DataFrame
        DataFrame com a coluna `high_quality` adicionada.
    """
    if "quality" not in df.columns:
        raise ValueError("A coluna 'quality' não foi encontrada no DataFrame.")

    df = df.copy()
    df[TARGET_COL] = np.where(df["quality"] >= quality_threshold, 1, 0)

    dist = df[TARGET_COL].value_counts(normalize=True) * 100
    print(f"[data_loader] Distribuição da variável alvo:")
    print(f"  Alta qualidade   (1): {dist.get(1, 0):.1f}%")
    print(f"  Baixa/Média qual (0): {dist.get(0, 0):.1f}%")

    return df


def split_features_target(df: pd.DataFrame):
    """
    Separa as variáveis explicativas (X) da variável alvo (y).

    Colunas removidas de X: `quality`, `high_quality` e `Id` (se existir).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com a coluna `high_quality` já criada.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) onde X contém as features e y contém o alvo.
    """
    cols_to_drop = [c for c in DROP_COLS + [TARGET_COL] if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COL]

    print(f"[data_loader] X: {X.shape} | y: {y.shape}")
    return X, y


def basic_info(df: pd.DataFrame) -> None:
    """
    Exibe informações básicas sobre o DataFrame: shape, tipos, nulos e estatísticas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser inspecionado.
    """
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print("-" * 60)
    print("Tipos de dados:")
    print(df.dtypes)
    print("-" * 60)
    print("Valores nulos:")
    print(df.isnull().sum())
    print("-" * 60)
    print("Estatísticas descritivas:")
    print(df.describe().T.round(3))
    print("=" * 60)
