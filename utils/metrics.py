"""
utils/metrics.py
================
Funções para calcular, exibir e salvar as métricas de avaliação dos modelos.

Utiliza scikit-learn como principal biblioteca de métricas, conforme requisito
do enunciado.

Métricas calculadas para cada experimento:
    - Accuracy
    - Precision (weighted)
    - Recall (weighted)
    - F1-Score (weighted)
    - Classification Report completo
    - Confusion Matrix
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Nomes das classes para exibição nos relatórios
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Caminho onde os resultados serão salvos
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics.csv")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula as quatro métricas principais para um experimento.

    Usa average='weighted' para lidar corretamente com múltiplas classes,
    dando peso proporcional ao número de amostras por classe.

    Args:
        y_true : Rótulos verdadeiros (inteiros).
        y_pred : Rótulos preditos pelo modelo (inteiros).

    Returns:
        Dicionário com as métricas: accuracy, precision, recall, f1.
    """
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, experiment_name: str = ""):
    """
    Exibe o relatório de classificação completo por classe (sklearn).

    Args:
        y_true          : Rótulos verdadeiros.
        y_pred          : Rótulos preditos.
        experiment_name : Nome do experimento para identificação no output.
    """
    if experiment_name:
        print(f"\n{'=' * 55}")
        print(f"  Experimento: {experiment_name}")
        print(f"{'=' * 55}")

    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        digits=4
    )
    print(report)
    return report


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de confusão.

    Args:
        y_true : Rótulos verdadeiros.
        y_pred : Rótulos preditos.

    Returns:
        Matriz de confusão (np.ndarray de shape 4x4).
    """
    return confusion_matrix(y_true, y_pred)


def save_metrics_to_csv(experiment_name: str, model_type: str, metrics: dict):
    """
    Salva (ou acumula) os resultados de um experimento em metrics.csv.

    Se o arquivo já existir, o novo resultado é adicionado ao final.
    Isso permite rodar experimentos separadamente e acumular os resultados.

    Args:
        experiment_name : Ex: "CNN Básica", "MLP + Dropout"
        model_type      : "CNN" ou "MLP"
        metrics         : Dicionário retornado por compute_metrics()
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    row = {
        "Experimento": experiment_name,
        "Modelo":      model_type,
        "Accuracy":    metrics["accuracy"],
        "Precision":   metrics["precision"],
        "Recall":      metrics["recall"],
        "F1":          metrics["f1"],
    }

    # Carrega CSV existente ou cria novo DataFrame
    if os.path.exists(METRICS_CSV):
        df = pd.read_csv(METRICS_CSV)
        # Atualiza linha se o experimento já existir
        if experiment_name in df["Experimento"].values:
            df.loc[df["Experimento"] == experiment_name, list(row.keys())] = list(row.values())
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(METRICS_CSV, index=False)
    print(f"  [✓] Métricas salvas em: results/metrics.csv")


def print_metrics_table():
    """
    Lê e exibe a tabela de métricas acumulada (todos os experimentos já rodados).
    """
    if not os.path.exists(METRICS_CSV):
        print("[AVISO] Nenhum resultado encontrado. Execute os experimentos primeiro.")
        return

    df = pd.read_csv(METRICS_CSV)
    print("\n" + "=" * 70)
    print("  TABELA DE RESULTADOS – TODOS OS EXPERIMENTOS")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")
