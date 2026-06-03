"""
evaluate.py
-----------
Funções para avaliação, comparação e visualização dos resultados dos modelos.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# ---------------------------------------------------------------------------
# Métricas consolidadas
# ---------------------------------------------------------------------------

def compute_metrics(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Calcula as principais métricas de classificação binária para um modelo.

    Parameters
    ----------
    model_name : str
        Nome do modelo para identificação na tabela de resultados.
    y_true : pd.Series
        Valores reais da variável alvo.
    y_pred : np.ndarray
        Predições binárias do modelo.
    y_proba : np.ndarray
        Probabilidades preditas para a classe positiva (classe 1).

    Returns
    -------
    dict
        Dicionário com as métricas calculadas.
    """
    return {
        "Modelo": model_name,
        "Acurácia":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "F1-score":  round(f1_score(y_true, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_proba), 4),
    }


def evaluate_all_models(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Avalia todos os modelos treinados e retorna uma tabela comparativa de métricas.

    Parameters
    ----------
    trained_models : dict
        Dicionário {nome: estimator_treinado}.
    X_test : pd.DataFrame
        Features do conjunto de teste.
    y_test : pd.Series
        Alvo real do conjunto de teste.

    Returns
    -------
    pd.DataFrame
        Tabela com as métricas de cada modelo, ordenada por F1-score.
    """
    results = []
    for name, model in trained_models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results.append(compute_metrics(name, y_test, y_pred, y_proba))

    df_results = pd.DataFrame(results).sort_values("F1-score", ascending=False).reset_index(drop=True)
    return df_results


# ---------------------------------------------------------------------------
# Relatório de classificação
# ---------------------------------------------------------------------------

def print_classification_reports(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Imprime o relatório de classificação completo de cada modelo.

    Parameters
    ----------
    trained_models : dict
        Dicionário {nome: estimator_treinado}.
    X_test : pd.DataFrame
        Features do conjunto de teste.
    y_test : pd.Series
        Alvo real do conjunto de teste.
    """
    labels = ["Baixa/Média", "Alta"]
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        print("=" * 60)
        print(f"Modelo: {name}")
        print("=" * 60)
        print(classification_report(y_test, y_pred, target_names=labels))


# ---------------------------------------------------------------------------
# Matriz de confusão
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """
    Plota a matriz de confusão de um modelo.

    Parameters
    ----------
    y_true : pd.Series
        Valores reais.
    y_pred : np.ndarray
        Predições do modelo.
    model_name : str
        Nome do modelo exibido no título.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Baixa/Média", "Alta"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Matriz de Confusão — {model_name}")
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.tight_layout()
    plt.show()


def plot_all_confusion_matrices(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Plota a matriz de confusão de todos os modelos.

    Parameters
    ----------
    trained_models : dict
        Dicionário {nome: estimator_treinado}.
    X_test : pd.DataFrame
        Features do conjunto de teste.
    y_test : pd.Series
        Alvo real do conjunto de teste.
    """
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, name)


# ---------------------------------------------------------------------------
# Curva ROC
# ---------------------------------------------------------------------------

def plot_roc_curves(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Plota a curva ROC de cada modelo em um único gráfico comparativo.

    Parameters
    ----------
    trained_models : dict
        Dicionário {nome: estimator_treinado}.
    X_test : pd.DataFrame
        Features do conjunto de teste.
    y_test : pd.Series
        Alvo real do conjunto de teste.
    """
    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=ax)

    plt.plot([0, 1], [0, 1], "k--", label="Classificador aleatório")
    plt.title("Curva ROC — Comparativo dos modelos")
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Gráfico comparativo de métricas
# ---------------------------------------------------------------------------

def plot_metrics_comparison(results_df: pd.DataFrame) -> None:
    """
    Plota um gráfico de barras comparando as métricas de todos os modelos.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame retornado por `evaluate_all_models()`.
    """
    results_melted = results_df.melt(
        id_vars="Modelo",
        var_name="Métrica",
        value_name="Valor",
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_melted, x="Métrica", y="Valor", hue="Modelo", palette="Set2")
    plt.title("Comparação das métricas dos modelos")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.legend(title="Modelo")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Importância de variáveis
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list, model_name: str = "Random Forest") -> pd.DataFrame:
    """
    Plota e retorna a importância das variáveis de um modelo baseado em árvores.

    Parameters
    ----------
    model : estimator
        Modelo treinado com o atributo `feature_importances_`.
    feature_names : list
        Lista com os nomes das variáveis.
    model_name : str, optional
        Nome do modelo exibido no título. Default: 'Random Forest'.

    Returns
    -------
    pd.DataFrame
        Tabela com variável e importância, ordenada de forma decrescente.

    Raises
    ------
    AttributeError
        Se o modelo não possuir o atributo `feature_importances_`.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"O modelo '{model_name}' não possui `feature_importances_`.")

    importance_df = pd.DataFrame({
        "Variável": feature_names,
        "Importância": model.feature_importances_,
    }).sort_values("Importância", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance_df, x="Importância", y="Variável", palette="Blues_r")
    plt.title(f"Importância das variáveis — {model_name}")
    plt.xlabel("Importância")
    plt.ylabel("Variável")
    plt.tight_layout()
    plt.show()

    return importance_df
