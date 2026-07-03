"""
evaluate.py
===========
Avalia todos os modelos treinados e gera os resultados finais do projeto.

Este script deve ser executado APÓS train_cnn.py e train_mlp.py.
Ele carrega cada modelo salvo em models/, realiza predições no conjunto
de teste e produz:

    - Tabela comparativa de métricas (results/metrics.csv)
    - Gráfico de comparação entre experimentos
    - Classification Report e Confusion Matrix de cada experimento

Uso:
    # Avaliar todos os experimentos:
    python evaluate.py

    # Avaliar apenas um experimento específico:
    python evaluate.py --exp 1

    # Só exibir a tabela de resultados já salvos:
    python evaluate.py --table
"""

import os
import argparse
import numpy as np
import tensorflow as tf

# Módulos do projeto
from experiments import ALL_EXPERIMENTS, get_experiment
from utils.data_loader import load_data_cnn, load_data_mlp
from utils.metrics import (
    compute_metrics,
    print_classification_report,
    save_metrics_to_csv,
    get_confusion_matrix,
    print_metrics_table,
)
from utils.plots import plot_confusion_matrix, plot_experiment_comparison

# ──────────────────────────────────────────────
# Configurações
# ──────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def get_model_path(experiment_name: str) -> str:
    """Retorna o caminho esperado do modelo salvo para um dado experimento."""
    safe_name = experiment_name.replace(" ", "_").replace("+", "plus")
    return os.path.join(MODELS_DIR, f"{safe_name}.keras")


def evaluate_experiment(config: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict | None:
    """
    Carrega o modelo treinado de um experimento e avalia no conjunto de teste.

    Etapas:
        1. Verifica se o arquivo do modelo existe
        2. Carrega o modelo salvo pelo ModelCheckpoint
        3. Realiza predições
        4. Calcula métricas com sklearn
        5. Exibe Classification Report
        6. Gera e salva Confusion Matrix
        7. Persiste métricas no CSV

    Args:
        config : Configuração do experimento (de experiments.py).
        X_test : Array de imagens/vetores de teste normalizados.
        y_test : Rótulos verdadeiros do conjunto de teste.

    Returns:
        Dicionário com as métricas, ou None se o modelo não foi encontrado.
    """
    exp_name   = config["name"]
    exp_id     = config["id"]
    model_path = get_model_path(exp_name)

    print(f"\n{'─' * 60}")
    print(f"  [{exp_id}] {exp_name}")
    print(f"{'─' * 60}")

    # Verifica se o modelo foi treinado
    if not os.path.exists(model_path):
        print(f"  [!] Modelo não encontrado: {model_path}")
        print(f"      Execute train_cnn.py ou train_mlp.py primeiro.")
        return None

    # Carrega o modelo
    print(f"  Carregando: {os.path.basename(model_path)}")
    model = tf.keras.models.load_model(model_path)

    # Predição
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    # Métricas
    metrics = compute_metrics(y_test, y_pred)
    cm      = get_confusion_matrix(y_test, y_pred)

    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")

    # Classification Report detalhado
    print_classification_report(y_test, y_pred, experiment_name=exp_name)

    # Confusion Matrix
    plot_confusion_matrix(cm, experiment_name=exp_name)

    # Salva no CSV
    save_metrics_to_csv(
        experiment_name=exp_name,
        model_type=config["model_type"],
        metrics=metrics,
    )

    return metrics


# ──────────────────────────────────────────────
# Ponto de entrada
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Avaliação dos experimentos treinados")
    parser.add_argument(
        "--exp",
        type=int,
        default=None,
        choices=list(range(1, 9)),
        help="ID do experimento a avaliar (1-8). Sem argumento: avalia todos.",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Apenas exibe a tabela de resultados já salvos em metrics.csv.",
    )
    args = parser.parse_args()

    # Modo tabela apenas
    if args.table:
        print_metrics_table()
        plot_experiment_comparison()
        return

    # Define quais experimentos avaliar
    if args.exp is not None:
        experiments = [get_experiment(args.exp)]
    else:
        experiments = ALL_EXPERIMENTS

    print("\n" + "=" * 60)
    print("  AVALIAÇÃO DOS EXPERIMENTOS")
    print("=" * 60)

    # ── Carrega dados de teste CNN (128×128) ──────
    # Carregamos uma vez para os experimentos CNN e outra para MLP
    cnn_experiments = [e for e in experiments if e["model_type"] == "CNN"]
    mlp_experiments = [e for e in experiments if e["model_type"] == "MLP"]

    all_results = []

    # ── Avalia experimentos CNN ───────────────────
    if cnn_experiments:
        print("\n  Carregando dados de teste para CNN (128×128)...")
        _, _, X_test_cnn, _, _, y_test_cnn = load_data_cnn(img_size=(128, 128))

        for config in cnn_experiments:
            metrics = evaluate_experiment(config, X_test_cnn, y_test_cnn)
            if metrics:
                all_results.append((config["name"], metrics))

    # ── Avalia experimentos MLP ───────────────────
    if mlp_experiments:
        print("\n  Carregando dados de teste para MLP (64×64)...")
        _, _, X_test_mlp, _, _, y_test_mlp = load_data_mlp(img_size=(64, 64))

        for config in mlp_experiments:
            metrics = evaluate_experiment(config, X_test_mlp, y_test_mlp)
            if metrics:
                all_results.append((config["name"], metrics))

    # ── Tabela final ──────────────────────────────
    if all_results:
        print_metrics_table()

        # Gráfico de comparação entre todos os experimentos avaliados
        print("\n  Gerando gráfico comparativo...")
        plot_experiment_comparison()

    print("\n" + "=" * 60)
    print("  AVALIAÇÃO CONCLUÍDA")
    print("  Resultados em: results/metrics.csv")
    print("  Gráficos em  : results/plots/ e results/confusion/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
