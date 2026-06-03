"""
train.py
--------
Funções para criação, treinamento e persistência dos modelos de classificação.
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Divisão treino / teste
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Divide os dados em conjuntos de treino e teste com estratificação.

    Parameters
    ----------
    X : pd.DataFrame
        Variáveis explicativas.
    y : pd.Series
        Variável alvo.
    test_size : float, optional
        Proporção reservada para teste. Default: 0.2.
    random_state : int, optional
        Semente para reprodutibilidade. Default: 42.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[train] Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
    print(f"  Distribuição no treino:\n{y_train.value_counts(normalize=True).mul(100).round(1).to_string()}")
    print(f"  Distribuição no teste:\n{y_test.value_counts(normalize=True).mul(100).round(1).to_string()}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Definição dos modelos
# ---------------------------------------------------------------------------

def build_models() -> dict:
    """
    Retorna um dicionário com os três modelos de classificação configurados.

    Os modelos incluídos são:
    - Regressão Logística (com StandardScaler via Pipeline)
    - Random Forest
    - Gradient Boosting

    Returns
    -------
    dict
        Dicionário {nome_do_modelo: estimator}.
    """
    models = {
        "Regressão Logística": Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }
    return models


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

def train_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict:
    """
    Treina todos os modelos recebidos e retorna os estimadores ajustados.

    Parameters
    ----------
    models : dict
        Dicionário {nome: estimator} retornado por `build_models()`.
    X_train : pd.DataFrame
        Features de treino.
    y_train : pd.Series
        Alvo de treino.

    Returns
    -------
    dict
        Dicionário {nome: estimator_treinado}.
    """
    trained = {}
    for name, model in models.items():
        print(f"[train] Treinando: {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"[train] {name} concluído.")

    return trained


# ---------------------------------------------------------------------------
# Validação cruzada
# ---------------------------------------------------------------------------

def cross_validate_models(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "f1",
) -> pd.DataFrame:
    """
    Realiza validação cruzada em todos os modelos e retorna um resumo comparativo.

    Parameters
    ----------
    models : dict
        Dicionário {nome: estimator} (pode ser treinado ou não treinado).
    X_train : pd.DataFrame
        Features de treino para cross-validation.
    y_train : pd.Series
        Alvo de treino para cross-validation.
    cv : int, optional
        Número de folds. Default: 5.
    scoring : str, optional
        Métrica usada na validação cruzada. Default: 'f1'.

    Returns
    -------
    pd.DataFrame
        Tabela com média e desvio-padrão do score por modelo.
    """
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        results.append({
            "Modelo": name,
            f"{scoring}_mean": scores.mean().round(4),
            f"{scoring}_std": scores.std().round(4),
        })
        print(f"[train] {name} | {scoring} CV: {scores.mean():.4f} ± {scores.std():.4f}")

    return pd.DataFrame(results).sort_values(f"{scoring}_mean", ascending=False)


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def save_model(model, name: str, output_dir: str = "models") -> str:
    """
    Salva um modelo treinado em disco usando pickle.

    Parameters
    ----------
    model : estimator
        Modelo scikit-learn já treinado.
    name : str
        Nome do modelo (usado para nomear o arquivo).
    output_dir : str, optional
        Diretório de destino. Default: 'models'.

    Returns
    -------
    str
        Caminho completo do arquivo salvo.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = name.lower().replace(" ", "_") + ".pkl"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    print(f"[train] Modelo salvo em: {filepath}")
    return filepath


def load_model(filepath: str):
    """
    Carrega um modelo salvo em disco.

    Parameters
    ----------
    filepath : str
        Caminho para o arquivo .pkl.

    Returns
    -------
    estimator
        Modelo scikit-learn carregado.
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)

    print(f"[train] Modelo carregado de: {filepath}")
    return model
