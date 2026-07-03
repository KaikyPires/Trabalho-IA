"""
utils/data_loader.py
====================
Responsável por carregar, pré-processar e normalizar as imagens do dataset
Brain Tumor MRI Dataset (masoudnickparvar - Kaggle).

Estrutura esperada do dataset:
    <DATASET_DIR>/
        Training/
            glioma/
            meningioma/
            notumor/
            pituitary/
        Testing/
            glioma/
            meningioma/
            notumor/
            pituitary/
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ──────────────────────────────────────────────
# Configurações globais
# ──────────────────────────────────────────────

# Caminho raiz do dataset (relativo ao projeto)
DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "kagglehub", "datasets", "masoudnickparvar",
    "brain-tumor-mri-dataset", "versions", "2"
)

# Classes do dataset (em ordem — serão os índices 0, 1, 2, 3)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Tamanho padrão das imagens para CNN
IMG_SIZE_CNN = (128, 128)

# Tamanho para MLP (menor para reduzir dimensionalidade do vetor achatado)
IMG_SIZE_MLP = (64, 64)


# ──────────────────────────────────────────────
# Funções auxiliares
# ──────────────────────────────────────────────

def _load_images_from_folder(folder: str, img_size: tuple, desc: str = "") -> tuple:
    """
    Carrega todas as imagens de uma pasta organizada por subpastas de classes.

    Args:
        folder   : Caminho da pasta (ex: .../Training)
        img_size : Tupla (largura, altura) para redimensionamento
        desc     : Texto exibido no progresso (tqdm)

    Returns:
        X (np.ndarray): Array de imagens com shape (N, H, W, 3), dtype float32
        y (np.ndarray): Array de rótulos inteiros com shape (N,)
    """
    images = []
    labels = []

    for label_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(folder, class_name)

        if not os.path.isdir(class_folder):
            raise FileNotFoundError(
                f"Pasta da classe '{class_name}' não encontrada em: {class_folder}"
            )

        file_list = [
            f for f in os.listdir(class_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for filename in tqdm(file_list, desc=f"{desc} [{class_name}]", leave=False):
            img_path = os.path.join(class_folder, filename)
            try:
                img = Image.open(img_path).convert("RGB")       # garante 3 canais
                img = img.resize(img_size, Image.LANCZOS)        # redimensiona
                images.append(np.array(img, dtype=np.float32))
                labels.append(label_idx)
            except Exception as e:
                print(f"[AVISO] Erro ao carregar {img_path}: {e}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def _normalize(X: np.ndarray) -> np.ndarray:
    """
    Normaliza os pixels para o intervalo [0, 1].

    Args:
        X: Array de imagens com valores entre 0 e 255.

    Returns:
        Array normalizado com valores entre 0.0 e 1.0.
    """
    return X / 255.0


# ──────────────────────────────────────────────
# Função principal de carregamento — CNN
# ──────────────────────────────────────────────

def load_data_cnn(img_size: tuple = IMG_SIZE_CNN, val_split: float = 0.2, seed: int = 42):
    """
    Carrega e prepara os dados para experimentos com CNN.

    - Imagens mantidas no formato (H, W, 3) — adequado para camadas Conv2D.
    - Conjunto de treino dividido em treino/validação.
    - Conjunto de teste vem diretamente da pasta 'Testing'.

    Args:
        img_size  : Dimensão das imagens (default: 128x128).
        val_split : Proporção do treino reservada para validação (default: 20%).
        seed      : Semente aleatória para reprodutibilidade.

    Returns:
        X_train, X_val, X_test (np.ndarray): Imagens normalizadas (float32).
        y_train, y_val, y_test (np.ndarray): Rótulos inteiros.
    """
    print("=" * 50)
    print("Carregando dados para CNN...")
    print(f"  Tamanho das imagens : {img_size}")
    print(f"  Divisão validação   : {int(val_split * 100)}%")
    print("=" * 50)

    # Carrega treino completo e teste
    X_all_train, y_all_train = _load_images_from_folder(
        os.path.join(DATASET_DIR, "Training"), img_size, desc="Treino"
    )
    X_test, y_test = _load_images_from_folder(
        os.path.join(DATASET_DIR, "Testing"), img_size, desc="Teste"
    )

    # Normalização
    X_all_train = _normalize(X_all_train)
    X_test      = _normalize(X_test)

    # Divisão treino / validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_all_train, y_all_train,
        test_size=val_split,
        random_state=seed,
        stratify=y_all_train   # mantém proporção das classes
    )

    print(f"\nDados carregados com sucesso!")
    print(f"  Treino      : {X_train.shape[0]} imagens")
    print(f"  Validação   : {X_val.shape[0]} imagens")
    print(f"  Teste       : {X_test.shape[0]} imagens")
    print(f"  Shape input : {X_train.shape[1:]}")
    print()

    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────
# Função principal de carregamento — MLP
# ──────────────────────────────────────────────

def load_data_mlp(img_size: tuple = IMG_SIZE_MLP, val_split: float = 0.2, seed: int = 42):
    """
    Carrega e prepara os dados para experimentos com MLP.

    - Imagens achatadas (flatten) em vetores 1D.
    - Tamanho menor (64x64) para reduzir a dimensão do vetor de entrada.
      (64 × 64 × 3 = 12.288 features — razoável para MLP)

    Args:
        img_size  : Dimensão das imagens (default: 64x64).
        val_split : Proporção do treino reservada para validação (default: 20%).
        seed      : Semente aleatória para reprodutibilidade.

    Returns:
        X_train, X_val, X_test (np.ndarray): Vetores 1D normalizados (float32).
        y_train, y_val, y_test (np.ndarray): Rótulos inteiros.
    """
    print("=" * 50)
    print("Carregando dados para MLP...")
    print(f"  Tamanho das imagens : {img_size}")
    print(f"  Divisão validação   : {int(val_split * 100)}%")
    print("=" * 50)

    # Reutiliza o loader de CNN com tamanho menor
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_cnn(
        img_size=img_size, val_split=val_split, seed=seed
    )

    # Achata imagens para vetores 1D: (N, H, W, 3) → (N, H*W*3)
    n_features = img_size[0] * img_size[1] * 3
    X_train = X_train.reshape(-1, n_features)
    X_val   = X_val.reshape(-1, n_features)
    X_test  = X_test.reshape(-1, n_features)

    print(f"  Shape MLP input : {X_train.shape[1:]}  ({n_features} features)\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────
# Informações do dataset (uso diagnóstico)
# ──────────────────────────────────────────────

def dataset_info():
    """
    Imprime um resumo do dataset: número de imagens por split e por classe.
    Útil para confirmar que os dados estão corretos antes de treinar.
    """
    print("=" * 50)
    print("Informações do Dataset")
    print("=" * 50)

    total = 0
    for split in ["Training", "Testing"]:
        split_path = os.path.join(DATASET_DIR, split)
        print(f"\n{split}:")
        for class_name in CLASSES:
            class_path = os.path.join(split_path, class_name)
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            total += count
            print(f"  {class_name:<15}: {count} imagens")

    print(f"\n  Total geral    : {total} imagens")
    print(f"  Classes        : {CLASSES}")
    print("=" * 50)


# ──────────────────────────────────────────────
# Teste rápido (executar diretamente: python utils/data_loader.py)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    dataset_info()

    print("\nTestando load_data_cnn()...")
    X_tr, X_v, X_te, y_tr, y_v, y_te = load_data_cnn()
    print(f"X_train: {X_tr.shape} | y_train: {y_tr.shape}")
    print(f"X_val  : {X_v.shape}  | y_val  : {y_v.shape}")
    print(f"X_test : {X_te.shape} | y_test : {y_te.shape}")

    print("\nTestando load_data_mlp()...")
    X_tr, X_v, X_te, y_tr, y_v, y_te = load_data_mlp()
    print(f"X_train: {X_tr.shape} | y_train: {y_tr.shape}")
    print(f"X_val  : {X_v.shape}  | y_val  : {y_v.shape}")
    print(f"X_test : {X_te.shape} | y_test : {y_te.shape}")
