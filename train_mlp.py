"""
train_mlp.py
============
Treina os 4 experimentos com MLP (Experimentos 5 a 8).

Cada experimento é configurado em experiments.py e executado aqui de forma
automática e sequencial. A função build_mlp() constrói dinamicamente cada
modelo com base nos hiperparâmetros recebidos.

Diferença principal em relação à CNN:
    - Entrada: vetor achatado (64×64×3 = 12.288 features)
    - Não há camadas convolucionais — apenas Dense layers
    - A MLP não captura relações espaciais da imagem, portanto é esperado
      desempenho inferior à CNN. Isso é pedagogicamente valioso para comparação.

Uso:
    # Treinar todos os 4 experimentos MLP:
    python train_mlp.py

    # Treinar apenas um experimento específico (ex: experimento 7):
    python train_mlp.py --exp 7
"""

import os
import unicodedata
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Módulos do projeto
from experiments import MLP_EXPERIMENTS, get_experiment
from utils.data_loader import load_data_mlp
from utils.metrics import compute_metrics, print_classification_report, save_metrics_to_csv, get_confusion_matrix
from utils.plots import plot_training_history, plot_confusion_matrix

# ──────────────────────────────────────────────
# Configurações gerais
# ──────────────────────────────────────────────
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
NUM_CLASSES = 4
SEED        = 42

# Garante reprodutibilidade
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────
# Construção dinâmica do modelo MLP
# ──────────────────────────────────────────────

def _sanitize_name(name: str) -> str:
    """Remove acentos e caracteres especiais para uso como nome de modelo no TensorFlow."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return name.replace(" ", "_").replace("+", "plus")


def build_mlp(config: dict) -> keras.Model:
    """
    Constrói um modelo MLP com base nas configurações do experimento.

    Arquitetura:
        Input(12.288)
        → Dense(mlp_layers[0], relu, [L2])
        → [Dropout(dropout_rate)]
        → Dense(mlp_layers[1], relu, [L2])
        → [Dropout(dropout_rate)]
        → ... (para cada camada em mlp_layers)
        → Dense(4, softmax)

    BatchNormalization é usada entre camadas para estabilizar o treino,
    o que é especialmente útil em MLPs com entrada de alta dimensão.

    Args:
        config : Dicionário de configuração do experimento (de experiments.py).

    Returns:
        Modelo Keras compilado.
    """
    # Prepara regularizador L2 (None se lambda = 0)
    l2 = regularizers.l2(config["l2_lambda"]) if config["l2_lambda"] > 0 else None

    # Dimensão de entrada: H × W × 3 (achatado)
    img_h, img_w = config["img_size"]
    n_features = img_h * img_w * 3

    inputs = keras.Input(shape=(n_features,), name="input")
    x = inputs

    # Camadas ocultas densas
    for i, units in enumerate(config["mlp_layers"]):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=l2,
            name=f"dense_{i+1}"
        )(x)
        # BatchNorm após ativação para estabilizar gradientes
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)

        # Dropout entre camadas (apenas se configurado)
        if config["dropout_rate"] > 0:
            x = layers.Dropout(config["dropout_rate"], name=f"drop_{i+1}")(x)

    # Camada de saída — softmax para classificação multiclasse
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=_sanitize_name(config["name"]))

    # Compilação com Adam e sparse_categorical_crossentropy
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ──────────────────────────────────────────────
# Treinamento de um experimento
# ──────────────────────────────────────────────

def train_experiment(config: dict, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Executa o ciclo completo de um experimento MLP:
        1. Constrói o modelo
        2. Exibe o resumo
        3. Treina com callbacks
        4. Avalia no conjunto de teste
        5. Salva modelo, gráficos e métricas

    Args:
        config                     : Dicionário de configuração do experimento.
        X_train, X_val, X_test     : Arrays de vetores normalizados (1D).
        y_train, y_val, y_test     : Arrays de rótulos inteiros.
    """
    exp_name = config["name"]
    exp_id   = config["id"]

    print(f"\n{'#' * 60}")
    print(f"  EXPERIMENTO {exp_id}: {exp_name}")
    print(f"{'#' * 60}")

    # ── 1. Construção do modelo ──────────────────
    model = build_mlp(config)
    model.summary()

    # ── 2. Callbacks ─────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_name  = exp_name.replace(" ", "_").replace("+", "plus")
    model_path = os.path.join(MODELS_DIR, f"{safe_name}.keras")

    callbacks = [
        # Para o treino se val_loss não melhorar por 5 épocas
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        # Salva o melhor modelo (menor val_loss)
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        # Reduz LR se val_loss estagnar por 3 épocas
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── 3. Treinamento ───────────────────────────
    print(f"\n  Treinando por até {config['epochs']} épocas...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    # ── 4. Avaliação no conjunto de teste ────────
    print(f"\n  Avaliando no conjunto de teste...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    # Métricas sklearn
    metrics = compute_metrics(y_test, y_pred)
    cm      = get_confusion_matrix(y_test, y_pred)

    print(f"\n  Resultados do Experimento {exp_id}:")
    print(f"    Accuracy  : {metrics['accuracy']:.4f}")
    print(f"    Precision : {metrics['precision']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    print(f"    F1-Score  : {metrics['f1']:.4f}")

    # Classification Report completo
    print_classification_report(y_test, y_pred, experiment_name=exp_name)

    # ── 5. Salvamento ────────────────────────────
    # Gráficos de treino (accuracy/loss por época)
    plot_training_history(history, experiment_name=exp_name)

    # Matriz de confusão
    plot_confusion_matrix(cm, experiment_name=exp_name)

    # Métricas no CSV acumulado
    save_metrics_to_csv(
        experiment_name=exp_name,
        model_type="MLP",
        metrics=metrics
    )

    print(f"\n  [✓] Modelo salvo: models/{safe_name}.keras")
    print(f"  [✓] Experimento {exp_id} concluído!\n")

    return metrics


# ──────────────────────────────────────────────
# Ponto de entrada
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Treinamento dos experimentos MLP")
    parser.add_argument(
        "--exp",
        type=int,
        default=None,
        choices=[5, 6, 7, 8],
        help="ID do experimento a executar (5-8). Sem argumento: executa todos."
    )
    args = parser.parse_args()

    # Define quais experimentos executar
    if args.exp is not None:
        experiments = [get_experiment(args.exp)]
    else:
        experiments = MLP_EXPERIMENTS

    # Carrega dados apenas uma vez — img_size do primeiro exp (todos usam 64x64)
    img_size = experiments[0]["img_size"]
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_mlp(img_size=img_size)

    # Executa cada experimento sequencialmente
    all_metrics = []
    for config in experiments:
        metrics = train_experiment(config, X_train, X_val, X_test, y_train, y_val, y_test)
        all_metrics.append((config["name"], metrics))

    # Resumo final dos experimentos executados nesta sessão
    print("\n" + "=" * 60)
    print("  RESUMO DOS EXPERIMENTOS MLP")
    print("=" * 60)
    print(f"  {'Experimento':<25} {'Acc':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("  " + "-" * 56)
    for name, m in all_metrics:
        print(
            f"  {name:<25} {m['accuracy']:>6.4f} {m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} {m['f1']:>7.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
