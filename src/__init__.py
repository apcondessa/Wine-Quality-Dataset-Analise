"""
src
---
Pacote auxiliar do projeto de classificação de qualidade de vinhos.

Módulos disponíveis
-------------------
- data_loader : carregamento, validação e pré-processamento inicial dos dados.
- eda         : análise exploratória e visualizações.
- train       : construção, treinamento e persistência dos modelos.
- evaluate    : métricas, gráficos de avaliação e importância de variáveis.
- pipeline    : orquestração do fluxo completo de ponta a ponta.
"""

from .data_loader import (
    NUMERIC_COLS,
    TARGET_COL,
    basic_info,
    create_binary_target,
    load_data,
    split_features_target,
)
from .eda import (
    compute_outliers_summary,
    plot_binary_target_distribution,
    plot_boxplots_by_class,
    plot_correlation_heatmap,
    plot_correlation_with_target,
    plot_feature_distributions,
    plot_outlier_boxplots,
    plot_quality_distribution,
)
from .evaluate import (
    compute_metrics,
    evaluate_all_models,
    plot_all_confusion_matrices,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metrics_comparison,
    plot_roc_curves,
    print_classification_reports,
)
from .train import (
    build_models,
    cross_validate_models,
    load_model,
    save_model,
    split_data,
    train_models,
)

__all__ = [
    # data_loader
    "NUMERIC_COLS", "TARGET_COL",
    "load_data", "create_binary_target", "split_features_target", "basic_info",
    # eda
    "plot_quality_distribution", "plot_binary_target_distribution",
    "plot_feature_distributions", "plot_boxplots_by_class",
    "plot_correlation_heatmap", "plot_correlation_with_target",
    "plot_outlier_boxplots", "compute_outliers_summary",
    # train
    "split_data", "build_models", "train_models",
    "cross_validate_models", "save_model", "load_model",
    # evaluate
    "compute_metrics", "evaluate_all_models", "print_classification_reports",
    "plot_confusion_matrix", "plot_all_confusion_matrices",
    "plot_roc_curves", "plot_metrics_comparison", "plot_feature_importance",
]
