# Classificação de Tumores Cerebrais com IA

**Projeto Prático – Inteligência Artificial 2026**  
IFMG – campus Ouro Branco – Bacharelado em Sistemas de Informação

---

## Descrição

Solução de classificação de imagens de ressonância magnética (RM) cerebral
utilizando técnicas de Aprendizado de Máquina. O projeto classifica imagens
em 4 categorias: **glioma**, **meningioma**, **pituitary (hipofisário)** e **sem tumor**.

**Dataset:** [Brain Tumor MRI Dataset – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
7.200 imagens | 4 classes balanceadas | Train: 5.600 | Test: 1.600

---

## Técnicas Utilizadas

| Técnica | Experimentos |
|---------|-------------|
| CNN (Convolutional Neural Network) | 1, 2, 3, 4 |
| MLP (Multilayer Perceptron) | 5, 6, 7, 8 |

---

## Experimentos

| # | Modelo | Variação |
|---|--------|----------|
| 1 | CNN Básica | Baseline sem regularização |
| 2 | CNN + Dropout | Dropout(0.3) nas conv + Dropout(0.5) no Dense |
| 3 | CNN + L2 | Regularização L2(0.001) |
| 4 | CNN Hiperparâmetros | Mais filtros, Dense maior, LR menor |
| 5 | MLP Básica | Baseline — Dense(256, 128) |
| 6 | MLP Mais Neurônios | Dense(512, 256, 128) |
| 7 | MLP + Dropout | Dropout(0.4) entre camadas |
| 8 | MLP + L2 | Regularização L2(0.001) + LR menor |

---

## Estrutura do Projeto

```
Trabalho/
│
├── kagglehub/               ← dataset local
├── models/                  ← modelos salvos (.keras)
├── results/
│   ├── plots/               ← gráficos accuracy/loss e comparação
│   ├── confusion/           ← matrizes de confusão
│   └── metrics.csv          ← tabela final de resultados
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       ← carregamento e pré-processamento
│   ├── metrics.py           ← cálculo de métricas (sklearn)
│   └── plots.py             ← geração de gráficos
│
├── experiments.py           ← configurações dos 8 experimentos
├── train_cnn.py             ← treina experimentos 1-4 (CNN)
├── train_mlp.py             ← treina experimentos 5-8 (MLP)
├── evaluate.py              ← avalia todos e gera resultados finais
├── requirements.txt
└── README.md
```

---

## Como Executar

> **Ambiente:** Use o Python do venv em `C:\tfenv\Scripts\python`

### 1. Treinar CNN (experimentos 1–4)

```bash
C:\tfenv\Scripts\python train_cnn.py
```

Treinar apenas um experimento:
```bash
C:\tfenv\Scripts\python train_cnn.py --exp 2
```

### 2. Treinar MLP (experimentos 5–8)

```bash
C:\tfenv\Scripts\python train_mlp.py
```

Treinar apenas um experimento:
```bash
C:\tfenv\Scripts\python train_mlp.py --exp 7
```

### 3. Avaliar e gerar resultados

```bash
C:\tfenv\Scripts\python evaluate.py
```

Apenas exibir tabela já salva:
```bash
C:\tfenv\Scripts\python evaluate.py --table
```

---

## Métricas

Calculadas com **scikit-learn** para todos os experimentos:

- **Accuracy** — proporção de acertos total
- **Precision** (weighted) — acurácia das predições positivas
- **Recall** (weighted) — cobertura das classes reais
- **F1-Score** (weighted) — média harmônica entre Precision e Recall

Além disso:
- **Classification Report** por classe
- **Confusion Matrix** (heatmap)

---

## Resultados

Após execução, os resultados ficam em `results/metrics.csv`:

| Experimento | Modelo | Accuracy | Precision | Recall | F1 |
|-------------|--------|----------|-----------|--------|----|
| CNN Básica | CNN | - | - | - | - |
| ... | ... | ... | ... | ... | ... |

---

## Tecnologias

- Python 3.12
- TensorFlow / Keras
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow
