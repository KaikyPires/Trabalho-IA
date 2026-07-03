"""
experiments.py
==============
Define as configurações de todos os 8 experimentos do projeto.

Cada experimento é um dicionário com todos os hiperparâmetros necessários
para construir e treinar o modelo correspondente. Centralizar as configurações
aqui evita duplicação de código em train_cnn.py e train_mlp.py.

Estrutura de cada experimento:
    - id            : Número do experimento (1 a 8)
    - name          : Nome descritivo (usado nos gráficos, relatório e CSV)
    - model_type    : "CNN" ou "MLP"
    - img_size      : Tamanho da imagem de entrada
    - epochs        : Número máximo de épocas (EarlyStopping pode interromper antes)
    - batch_size    : Tamanho do batch
    - learning_rate : Taxa de aprendizado do otimizador Adam
    - dropout_rate  : Taxa de dropout (0.0 = sem dropout)
    - l2_lambda     : Fator de regularização L2 (0.0 = sem regularização)
    - filters       : [CNN] Lista com número de filtros por bloco convolucional
    - dense_units   : Número de neurônios na(s) camada(s) densa(s)
    - mlp_layers    : [MLP] Lista com número de neurônios por camada oculta
"""

# ──────────────────────────────────────────────────────────────────────────────
# Experimentos CNN
# ──────────────────────────────────────────────────────────────────────────────
# Arquitetura base CNN:
#   Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
#   → Flatten → Dense(128) → Dense(4, softmax)
#
# Variações controladas entre experimentos:
#   Exp 1: baseline sem regularização
#   Exp 2: adiciona Dropout após cada bloco conv e antes do Dense
#   Exp 3: adiciona L2 nas camadas Conv e Dense (sem Dropout)
#   Exp 4: muda filtros, neurônios, epochs e learning_rate

CNN_EXPERIMENTS = [
    {
        # ──────────────────────────────────────────
        # Experimento 1 – CNN Básica (baseline)
        # Modelo mais simples possível. Sem regularização.
        # Serve como referência para comparar os demais.
        # ──────────────────────────────────────────
        "id":            1,
        "name":          "CNN Básica",
        "model_type":    "CNN",
        "img_size":      (128, 128),
        "epochs":        15,
        "batch_size":    32,
        "learning_rate": 0.001,
        "dropout_rate":  0.0,     # sem dropout
        "l2_lambda":     0.0,     # sem L2
        "filters":       [32, 64, 128],
        "dense_units":   128,
    },
    {
        # ──────────────────────────────────────────
        # Experimento 2 – CNN + Dropout
        # Técnica de regularização mais comum em CNNs.
        # Dropout(0.3) após cada MaxPool e Dropout(0.5) antes do Dense.
        # Objetivo: reduzir overfitting. Esperamos maior gap entre
        # val_accuracy e train_accuracy menor — modelo mais generalista.
        # ──────────────────────────────────────────
        "id":            2,
        "name":          "CNN + Dropout",
        "model_type":    "CNN",
        "img_size":      (128, 128),
        "epochs":        15,
        "batch_size":    32,
        "learning_rate": 0.001,
        "dropout_rate":  0.3,     # Dropout(0.3) nas conv + Dropout(0.5) no dense
        "l2_lambda":     0.0,
        "filters":       [32, 64, 128],
        "dense_units":   128,
    },
    {
        # ──────────────────────────────────────────
        # Experimento 3 – CNN + Regularização L2
        # L2 penaliza pesos grandes, forçando o modelo a distribuir
        # a importância entre mais features. Alternativa ao Dropout.
        # Objetivo: comparar L2 vs Dropout como estratégia anti-overfitting.
        # ──────────────────────────────────────────
        "id":            3,
        "name":          "CNN + L2",
        "model_type":    "CNN",
        "img_size":      (128, 128),
        "epochs":        15,
        "batch_size":    32,
        "learning_rate": 0.001,
        "dropout_rate":  0.0,
        "l2_lambda":     0.001,   # L2 em todas as camadas Conv e Dense
        "filters":       [32, 64, 128],
        "dense_units":   128,
    },
    {
        # ──────────────────────────────────────────
        # Experimento 4 – CNN com Hiperparâmetros Ajustados
        # Aumenta capacidade do modelo: mais filtros, camada densa maior.
        # Reduz learning rate e aumenta épocas para convergência mais fina.
        # Objetivo: verificar se maior capacidade melhora as métricas.
        # ──────────────────────────────────────────
        "id":            4,
        "name":          "CNN Hiperparâmetros",
        "model_type":    "CNN",
        "img_size":      (128, 128),
        "epochs":        20,
        "batch_size":    32,
        "learning_rate": 0.0005,  # learning rate menor → convergência mais estável
        "dropout_rate":  0.0,
        "l2_lambda":     0.0,
        "filters":       [64, 128, 256],  # dobra os filtros em relação ao baseline
        "dense_units":   256,             # camada densa maior
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Experimentos MLP
# ──────────────────────────────────────────────────────────────────────────────
# Entrada: vetor achatado de 64×64×3 = 12.288 features (normalizado [0,1])
# Saída  : 4 neurônios com softmax (uma por classe)
#
# Variações controladas:
#   Exp 5: MLP básica (baseline)
#   Exp 6: aumenta neurônios em todas as camadas
#   Exp 7: adiciona Dropout entre as camadas ocultas
#   Exp 8: adiciona L2 + reduz learning rate

MLP_EXPERIMENTS = [
    {
        # ──────────────────────────────────────────
        # Experimento 5 – MLP Básica (baseline)
        # Rede simples com duas camadas ocultas.
        # Serve como referência para os experimentos MLP.
        # Esperamos desempenho inferior à CNN (MLP não captura
        # relações espaciais das imagens).
        # ──────────────────────────────────────────
        "id":            5,
        "name":          "MLP Básica",
        "model_type":    "MLP",
        "img_size":      (64, 64),
        "epochs":        20,
        "batch_size":    32,
        "learning_rate": 0.001,
        "dropout_rate":  0.0,
        "l2_lambda":     0.0,
        "mlp_layers":    [256, 128],   # duas camadas ocultas
    },
    {
        # ──────────────────────────────────────────
        # Experimento 6 – MLP com Mais Neurônios
        # Aumenta capacidade da rede adicionando mais neurônios
        # e uma camada extra. Objetivo: verificar se maior capacidade
        # melhora a acurácia. Risco: overfitting.
        # ──────────────────────────────────────────
        "id":            6,
        "name":          "MLP Mais Neurônios",
        "model_type":    "MLP",
        "img_size":      (64, 64),
        "epochs":        20,
        "batch_size":    32,
        "learning_rate": 0.001,
        "dropout_rate":  0.0,
        "l2_lambda":     0.0,
        "mlp_layers":    [512, 256, 128],  # mais neurônios e camada extra
    },
    {
        # ──────────────────────────────────────────
        # Experimento 7 – MLP + Dropout
        # Dropout(0.4) entre cada camada oculta.
        # Objetivo: reduzir overfitting observado no exp. 6.
        # ──────────────────────────────────────────
        "id":            7,
        "name":          "MLP + Dropout",
        "model_type":    "MLP",
        "img_size":      (64, 64),
        "epochs":        20,
        "batch_size":    32,
        "learning_rate": 0.001,
        "dropout_rate":  0.4,
        "l2_lambda":     0.0,
        "mlp_layers":    [256, 128],
    },
    {
        # ──────────────────────────────────────────
        # Experimento 8 – MLP + Regularização L2
        # L2 nas camadas densas + learning rate menor para
        # treinamento mais estável com regularização.
        # Objetivo: comparar L2 vs Dropout na MLP, assim como
        # foi feito na CNN (exps. 2 e 3).
        # ──────────────────────────────────────────
        "id":            8,
        "name":          "MLP + L2",
        "model_type":    "MLP",
        "img_size":      (64, 64),
        "epochs":        20,
        "batch_size":    32,
        "learning_rate": 0.0005,
        "dropout_rate":  0.0,
        "l2_lambda":     0.001,
        "mlp_layers":    [256, 128],
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Lista unificada com todos os experimentos (CNN + MLP)
# Usada em evaluate.py para iterar sobre todos os resultados
# ──────────────────────────────────────────────────────────────────────────────
ALL_EXPERIMENTS = CNN_EXPERIMENTS + MLP_EXPERIMENTS


def get_experiment(experiment_id: int) -> dict:
    """
    Retorna a configuração de um experimento pelo seu ID (1 a 8).

    Args:
        experiment_id : ID do experimento.

    Returns:
        Dicionário de configuração do experimento.

    Raises:
        ValueError se o ID não for encontrado.
    """
    for exp in ALL_EXPERIMENTS:
        if exp["id"] == experiment_id:
            return exp
    raise ValueError(f"Experimento com ID {experiment_id} não encontrado.")


def print_experiments_summary():
    """
    Exibe um resumo formatado de todos os experimentos configurados.
    Útil para conferir as configurações antes de iniciar o treino.
    """
    print("\n" + "=" * 70)
    print("  CONFIGURAÇÃO DOS 8 EXPERIMENTOS")
    print("=" * 70)

    for exp in ALL_EXPERIMENTS:
        print(f"\n  [{exp['id']}] {exp['name']}  ({exp['model_type']})")
        print(f"       Épocas     : {exp['epochs']}")
        print(f"       Batch size : {exp['batch_size']}")
        print(f"       LR         : {exp['learning_rate']}")
        print(f"       Dropout    : {exp['dropout_rate']}")
        print(f"       L2 lambda  : {exp['l2_lambda']}")
        if exp["model_type"] == "CNN":
            print(f"       Filtros    : {exp['filters']}")
            print(f"       Dense      : {exp['dense_units']}")
        else:
            print(f"       Camadas    : {exp['mlp_layers']}")

    print("\n" + "=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Execução direta para conferir as configurações
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_experiments_summary()
