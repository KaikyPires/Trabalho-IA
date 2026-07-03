"""
utils/plots.py
==============
Funções para geração e salvamento de todos os gráficos do projeto:

    1. Curvas de accuracy por época (treino e validação)
    2. Curvas de loss por época (treino e validação)
    3. Matriz de confusão (heatmap)
    4. Gráfico comparativo de métricas entre todos os experimentos

Todos os gráficos são salvos automaticamente em results/plots/ ou
results/confusion/ e exibidos em tela apenas se show=True.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Diretórios de saída
BASE_DIR      = os.path.dirname(os.path.dirname(__file__))
PLOTS_DIR     = os.path.join(BASE_DIR, "results", "plots")
CONFUSION_DIR = os.path.join(BASE_DIR, "results", "confusion")
METRICS_CSV   = os.path.join(BASE_DIR, "results", "metrics.csv")

# Classes do dataset
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Estilo visual consistente em todos os gráficos
plt.rcParams.update({
    "figure.dpi":      120,
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _ensure_dirs():
    """Garante que os diretórios de saída existam."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(CONFUSION_DIR, exist_ok=True)


def plot_training_history(history, experiment_name: str, show: bool = False):
    """
    Gera e salva dois gráficos lado a lado:
        - Accuracy por época (treino e validação)
        - Loss por época (treino e validação)

    Args:
        history         : Objeto History retornado por model.fit() do Keras.
        experiment_name : Nome do experimento (usado no título e nome do arquivo).
        show            : Se True, exibe o gráfico na tela além de salvar.
    """
    _ensure_dirs()

    hist = history.history
    epochs = range(1, len(hist["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Histórico de Treino – {experiment_name}", fontsize=13, fontweight="bold")

    # ── Accuracy ──────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(epochs, hist["accuracy"],     label="Treino",    color="#4C72B0", linewidth=2)
    ax1.plot(epochs, hist["val_accuracy"], label="Validação", color="#DD8452", linewidth=2, linestyle="--")
    ax1.set_title("Accuracy por Época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ── Loss ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(epochs, hist["loss"],     label="Treino",    color="#4C72B0", linewidth=2)
    ax2.plot(epochs, hist["val_loss"], label="Validação", color="#DD8452", linewidth=2, linestyle="--")
    ax2.set_title("Loss por Época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()

    # Salva o arquivo
    safe_name = experiment_name.replace(" ", "_").replace("+", "plus")
    filepath = os.path.join(PLOTS_DIR, f"{safe_name}_history.png")
    plt.savefig(filepath, bbox_inches="tight")
    print(f"  [✓] Gráfico salvo: results/plots/{os.path.basename(filepath)}")

    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, experiment_name: str, show: bool = False):
    """
    Gera e salva a matriz de confusão como heatmap (seaborn).

    Args:
        cm              : Matriz de confusão (numpy array 4x4).
        experiment_name : Nome do experimento.
        show            : Se True, exibe o gráfico na tela além de salvar.
    """
    _ensure_dirs()

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(f"Matriz de Confusão – {experiment_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predito", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    safe_name = experiment_name.replace(" ", "_").replace("+", "plus")
    filepath = os.path.join(CONFUSION_DIR, f"{safe_name}_confusion.png")
    plt.savefig(filepath, bbox_inches="tight")
    print(f"  [✓] Matriz salva   : results/confusion/{os.path.basename(filepath)}")

    if show:
        plt.show()
    plt.close()


def plot_experiment_comparison(show: bool = False):
    """
    Lê o arquivo metrics.csv e gera um gráfico de barras comparando
    todos os experimentos nas 4 métricas (Accuracy, Precision, Recall, F1).

    Deve ser chamado após todos os experimentos terem sido rodados.

    Args:
        show : Se True, exibe o gráfico na tela além de salvar.
    """
    _ensure_dirs()

    if not os.path.exists(METRICS_CSV):
        print("[AVISO] metrics.csv não encontrado. Execute os experimentos antes.")
        return

    df = pd.read_csv(METRICS_CSV)

    if df.empty:
        print("[AVISO] Nenhum resultado encontrado em metrics.csv.")
        return

    metrics_cols = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(df))
    width = 0.18
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(max(12, len(df) * 1.5), 6))

    for i, (col, color) in enumerate(zip(metrics_cols, colors)):
        bars = ax.bar(x + i * width, df[col], width, label=col, color=color, alpha=0.85)
        # Valor numérico em cima de cada barra
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90
            )

    ax.set_title("Comparação entre Experimentos", fontsize=13, fontweight="bold")
    ax.set_xlabel("Experimento", fontsize=11)
    ax.set_ylabel("Valor da Métrica", fontsize=11)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["Experimento"], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Linha de referência em 0.80 e 0.90
    for ref in [0.80, 0.90]:
        ax.axhline(ref, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.text(len(df) - 0.3, ref + 0.005, f"{ref:.0%}", color="gray", fontsize=8)

    plt.tight_layout()

    filepath = os.path.join(PLOTS_DIR, "comparacao_experimentos.png")
    plt.savefig(filepath, bbox_inches="tight")
    print(f"\n  [✓] Comparação salva: results/plots/comparacao_experimentos.png")

    if show:
        plt.show()
    plt.close()
