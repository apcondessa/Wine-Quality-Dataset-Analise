"""
pipeline.py
-----------
Script principal que orquestra todo o fluxo do projeto:
  1. Carregamento dos dados
  2. Criação da variável alvo
  3. Análise exploratória
  4. Separação treino/teste
  5. Treinamento dos modelos
  6. Avaliação e comparação
  7. Salvamento dos modelos

Uso:
    python pipeline.py --data_path <caminho_do_csv>

Exemplo:
    python pipeline.py --data_path ../data/WineQT.csv
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

from data_loader import basic_info, create_binary_target, load_data, split_features_target
from eda import (
    compute_outliers_summary,
    plot_binary_target_distribution,
    plot_correlation_heatmap,
    plot_correlation_with_target,
    plot_feature_distributions,
    plot_quality_distribution,
)
from evaluate import (
    evaluate_all_models,
    plot_all_confusion_matrices,
    plot_feature_importance,
    plot_metrics_comparison,
    plot_roc_curves,
    print_classification_reports,
)
from train import build_models, cross_validate_models, save_model, split_data, train_models


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(data_path: str, run_eda: bool = True, save_models: bool = True) -> dict:
    """
    Executa o pipeline completo de classificação de qualidade de vinhos.

    Parameters
    ----------
    data_path : str
        Caminho para o arquivo CSV com os dados.
    run_eda : bool, optional
        Se True, executa a análise exploratória com gráficos. Default: True.
    save_models : bool, optional
        Se True, salva os modelos treinados em disco. Default: True.

    Returns
    -------
    dict
        Dicionário com os modelos treinados e o DataFrame de resultados.
    """

    # ------------------------------------------------------------------
    # 1. Carregamento e inspeção inicial
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPA 1 — Carregamento dos dados")
    print("=" * 60)
    df = load_data(data_path)
    basic_info(df)

    # ------------------------------------------------------------------
    # 2. Variável alvo binária
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPA 2 — Criação da variável alvo")
    print("=" * 60)
    df = create_binary_target(df, quality_threshold=7)

    # ------------------------------------------------------------------
    # 3. Análise exploratória
    # ------------------------------------------------------------------
    if run_eda:
        print("\n" + "=" * 60)
        print("ETAPA 3 — Análise exploratória de dados")
        print("=" * 60)
        plot_quality_distribution(df)
        plot_binary_target_distribution(df)
        plot_feature_distributions(df)
        plot_correlation_heatmap(df)
        plot_correlation_with_target(df)

        print("\n[pipeline] Resumo de outliers:")
        outliers_df = compute_outliers_summary(df)
        print(outliers_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Separação features / alvo e treino / teste
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPA 4 — Separação dos dados")
    print("=" * 60)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ------------------------------------------------------------------
    # 5. Validação cruzada e treinamento
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPA 5 — Validação cruzada e treinamento")
    print("=" * 60)
    models = build_models()

    print("\n[pipeline] Validação cruzada (5-fold, F1-score):")
    cv_results = cross_validate_models(models, X_train, y_train)
    print(cv_results.to_string(index=False))

    trained_models = train_models(models, X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Avaliação
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPA 6 — Avaliação dos modelos")
    print("=" * 60)
    results_df = evaluate_all_models(trained_models, X_test, y_test)
    print(results_df.to_string(index=False))

    print_classification_reports(trained_models, X_test, y_test)
    plot_all_confusion_matrices(trained_models, X_test, y_test)
    plot_roc_curves(trained_models, X_test, y_test)
    plot_metrics_comparison(results_df)

    # Importância de variáveis do Random Forest
    if "Random Forest" in trained_models:
        plot_feature_importance(
            trained_models["Random Forest"],
            feature_names=list(X.columns),
            model_name="Random Forest",
        )

    # ------------------------------------------------------------------
    # 7. Salvamento dos modelos
    # ------------------------------------------------------------------
    if save_models:
        print("\n" + "=" * 60)
        print("ETAPA 7 — Salvamento dos modelos")
        print("=" * 60)
        for name, model in trained_models.items():
            save_model(model, name, output_dir="models")

    print("\n[pipeline] Pipeline concluído com sucesso!")
    return {"models": trained_models, "results": results_df}


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de classificação de qualidade de vinhos."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Caminho para o arquivo WineQT.csv",
    )
    parser.add_argument(
        "--skip_eda",
        action="store_true",
        help="Se informado, pula a etapa de análise exploratória.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Se informado, não salva os modelos em disco.",
    )

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        run_eda=not args.skip_eda,
        save_models=not args.no_save,
    )
