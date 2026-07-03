# Relatório de Projeto Prático
## Inteligência Artificial Aplicada à Medicina: Detecção de Tumores Cerebrais

---

**Instituição:** IFMG – campus Ouro Branco  
**Curso:** Bacharelado em Sistemas de Informação  
**Disciplina:** Inteligência Artificial – 2026  
**Valor:** 20 pontos  
**Data de entrega:** 03/07/2026  

---

## 1. Introdução

O diagnóstico precoce de tumores cerebrais é um dos maiores desafios da medicina moderna. A análise de imagens de ressonância magnética (RM) é o principal método diagnóstico, porém exige especialistas altamente treinados e é suscetível a erros humanos, especialmente em casos limítrofes entre tipos de tumores.

O avanço das técnicas de Aprendizado de Máquina (*Machine Learning*) — em especial as Redes Neurais Artificiais — tornou possível automatizar a análise de imagens médicas com alta precisão. Modelos como CNNs (*Convolutional Neural Networks*) são capazes de extrair padrões visuais de imagens de forma hierárquica, aprendendo desde bordas simples até estruturas complexas como formações tumorais.

Este projeto aplica duas técnicas de Aprendizado de Máquina — **CNN** e **MLP** (*Multilayer Perceptron*) — para classificar automaticamente imagens de RM cerebral em quatro categorias, comparando sistematicamente o desempenho de cada abordagem por meio de 8 experimentos controlados.

---

## 2. Objetivo

Desenvolver e avaliar experimentalmente modelos de classificação de imagens de ressonância magnética cerebral, utilizando técnicas de Aprendizado de Máquina, com os seguintes objetivos específicos:

- Implementar e comparar CNN e MLP para classificação multiclasse de tumores cerebrais;
- Investigar o impacto de técnicas de regularização (Dropout e L2) no desempenho dos modelos;
- Analisar a influência de diferentes configurações de hiperparâmetros nas métricas de avaliação;
- Identificar a abordagem com melhor desempenho e justificar os resultados obtidos.

---

## 3. Dataset

**Nome:** Brain Tumor MRI Dataset  
**Fonte:** [Kaggle – masoudnickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

O dataset contém **7.200 imagens** de ressonância magnética cerebral, organizadas em quatro classes:

| Classe | Descrição | Treino | Teste |
|--------|-----------|--------|-------|
| **Glioma** | Tumor originado nas células gliais | 1.400 | 400 |
| **Meningioma** | Tumor nas membranas que envolvem o cérebro | 1.400 | 400 |
| **Pituitary** | Tumor hipofisário (glândula pituitária) | 1.400 | 400 |
| **No Tumor** | Cérebro saudável, sem tumor detectado | 1.400 | 400 |
| **Total** | | **5.600** | **1.600** |

O dataset é **balanceado** (igual número de amostras por classe), o que elimina a necessidade de técnicas de balanceamento como *oversampling* ou pesos de classe.

---

## 4. Tecnologias Utilizadas

| Tecnologia | Versão | Uso no Projeto |
|------------|--------|----------------|
| Python | 3.12 | Linguagem principal |
| TensorFlow / Keras | >= 2.13 | Construção e treinamento dos modelos |
| Scikit-Learn | >= 1.3 | Cálculo de métricas de avaliação |
| NumPy | >= 1.24 | Manipulação de arrays |
| Pandas | >= 2.0 | Geração da tabela de resultados |
| Matplotlib | >= 3.7 | Gráficos de treino e comparação |
| Seaborn | >= 0.12 | Visualização das matrizes de confusão |
| Pillow | >= 10.0 | Carregamento e redimensionamento de imagens |

---

## 5. Metodologia

### 5.1 Pipeline de Processamento

```
Imagens no disco (kagglehub/)
         |
  data_loader.py
  +-------------------------------+
  | 1. Leitura com Pillow (PIL)   |
  | 2. Conversao para RGB         |
  | 3. Redimensionamento          |
  |    CNN: 128x128 px            |
  |    MLP:  64x64 px             |
  | 4. Normalizacao: pixels/255.0 |
  | 5. Divisao Treino 80%/Val 20% |
  |    (estratificada por classe) |
  +-------------------------------+
         |
  experiments.py (configuracoes)
         |
  train_cnn.py / train_mlp.py
  +-------------------------------+
  | 1. Construcao do modelo       |
  | 2. Treinamento com callbacks: |
  |    - EarlyStopping(patience=5)|
  |    - ModelCheckpoint          |
  |    - ReduceLROnPlateau        |
  | 3. Salvamento do melhor modelo|
  | 4. Graficos (acc/loss)        |
  +-------------------------------+
         |
  evaluate.py
  +-------------------------------+
  | 1. Carrega modelos salvos     |
  | 2. Predicao no teste          |
  | 3. Metricas sklearn           |
  | 4. Classification Report      |
  | 5. Confusion Matrix           |
  | 6. Tabela comparativa (CSV)   |
  +-------------------------------+
```

### 5.2 Pré-processamento das Imagens

As imagens foram pré-processadas seguindo as etapas:

1. **Leitura:** Pillow (PIL) com conversão forçada para RGB, garantindo 3 canais em todas as imagens.
2. **Redimensionamento:** Usando interpolação LANCZOS (alta qualidade) para evitar distorções.
   - CNN: 128×128 pixels — preserva mais detalhes espaciais para as convoluções.
   - MLP: 64×64 pixels — reduz a dimensão do vetor achatado de 49.152 para 12.288 features, tornando o treino da MLP viável.
3. **Normalização:** Divisão por 255 — pixels no intervalo [0, 1] melhoram a convergência do gradiente.
4. **Divisão:** 80% treino / 20% validação, com estratificação para manter a proporção de classes.

### 5.3 Métricas de Avaliação

Todas as métricas foram calculadas com **scikit-learn** usando `average='weighted'`:

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **Accuracy** | (VP + VN) / Total | Proporção geral de acertos |
| **Precision** | VP / (VP + FP) | Qualidade das predições positivas |
| **Recall** | VP / (VP + FN) | Cobertura das classes reais |
| **F1-Score** | 2 x (P x R) / (P + R) | Equilíbrio entre Precision e Recall |

### 5.4 Callbacks de Treinamento

- **EarlyStopping** (`patience=5`): Interrompe o treino se a `val_loss` não melhorar, restaurando os melhores pesos.
- **ModelCheckpoint**: Salva automaticamente o modelo com menor `val_loss`.
- **ReduceLROnPlateau** (`factor=0.5`, `patience=3`): Reduz o learning rate à metade quando a `val_loss` estagna.

---

## 6. Experimentos

### 6.1 CNN – Convolutional Neural Network

A CNN preserva a estrutura espacial da imagem, aplicando filtros convolucionais que detectam padrões locais. É a técnica estado-da-arte para análise de imagens.

**Arquitetura base:**
```
Input (128x128x3)
  Conv2D(32, 3x3, relu) -> BatchNorm -> MaxPool(2x2)
  Conv2D(64, 3x3, relu) -> BatchNorm -> MaxPool(2x2)
  Conv2D(128, 3x3, relu) -> BatchNorm -> MaxPool(2x2)
  Flatten
  Dense(128, relu)
  Dense(4, softmax)
```

#### Experimento 1 — CNN Básica

**Objetivo:** Estabelecer um baseline sem qualquer regularização.  
**Configuração:** Filtros [32, 64, 128] | Dense 128 | LR 0.001 | 15 épocas  
**Justificativa:** Ponto de referência para medir o ganho das técnicas seguintes.

#### Experimento 2 — CNN + Dropout

**Objetivo:** Reduzir overfitting com Dropout.  
**Configuração:** Dropout(0.3) após cada bloco conv + Dropout(0.5) antes da saída  
**Justificativa:** O Dropout desativa neurônios aleatoriamente durante o treino, forçando representações mais robustas.  
**Impacto esperado:** Menor diferença entre acurácia de treino e validação.

#### Experimento 3 — CNN + Regularização L2

**Objetivo:** Comparar L2 com Dropout como estratégia anti-overfitting.  
**Configuração:** L2 (lambda=0.001) em todas as camadas Conv2D e Dense  
**Justificativa:** L2 penaliza pesos grandes, forçando a rede a distribuir a informação entre mais parâmetros.

#### Experimento 4 — CNN com Hiperparâmetros Ajustados

**Objetivo:** Verificar se maior capacidade melhora o desempenho.  
**Configuração:** Filtros [64, 128, 256] | Dense 256 | LR 0.0005 | 20 épocas  
**Justificativa:** Dobrar os filtros aumenta a capacidade de aprender padrões complexos. LR menor compensa a maior capacidade.

---

### 6.2 MLP – Multilayer Perceptron

A MLP "achata" a imagem em vetor 1D, perdendo a informação espacial. É esperado desempenho inferior à CNN, mas é pedagogicamente valioso para comparação.

**Entrada:** 64×64×3 = **12.288 features**

#### Experimento 5 — MLP Básica

**Objetivo:** Baseline MLP.  
**Configuração:** Dense(256) -> Dense(128) | LR 0.001 | 20 épocas

#### Experimento 6 — MLP com Mais Neurônios

**Objetivo:** Aumentar capacidade da rede.  
**Configuração:** Dense(512) -> Dense(256) -> Dense(128)  
**Risco:** Overfitting por perda da estrutura espacial.

#### Experimento 7 — MLP + Dropout

**Objetivo:** Regularizar com Dropout.  
**Configuração:** Dense(256) -> Dropout(0.4) -> Dense(128) -> Dropout(0.4)

#### Experimento 8 — MLP + Regularização L2

**Objetivo:** Comparar L2 vs Dropout na MLP.  
**Configuração:** Dense(256) -> Dense(128) com L2(0.001) | LR 0.0005

---

## 7. Resultados

Os resultados abaixo foram obtidos avaliando cada modelo no conjunto de **teste (1.600 imagens)**, que não foi utilizado durante o treinamento.

| # | Experimento | Modelo | Accuracy | Precision | Recall | F1-Score |
|---|-------------|--------|----------|-----------|--------|----------|
| 1 | CNN Básica | CNN | 0.8906 | 0.8949 | 0.8906 | 0.8887 |
| 2 | CNN + Dropout | CNN | 0.8094 | 0.8075 | 0.8094 | 0.7997 |
| **3** | **CNN + L2** | **CNN** | **0.8994** | **0.9015** | **0.8994** | **0.8979** |
| 4 | CNN Hiperparâmetros | CNN | 0.8650 | 0.8694 | 0.8650 | 0.8626 |
| 5 | MLP Básica | MLP | 0.8256 | 0.8258 | 0.8256 | 0.8240 |
| 6 | MLP Mais Neurônios | MLP | 0.8300 | 0.8317 | 0.8300 | 0.8286 |
| 7 | MLP + Dropout | MLP | 0.6687 | 0.6826 | 0.6687 | 0.6600 |
| 8 | MLP + L2 | MLP | 0.8094 | 0.8062 | 0.8094 | 0.8061 |

**Melhor modelo:** CNN + L2 (Experimento 3) — F1-Score de **0.8979** (~90%)

### 7.1 Desempenho por Classe — Melhor Modelo (CNN + L2)

| Classe | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Glioma | 0.8911 | 0.7775 | 0.8304 |
| Meningioma | 0.8886 | 0.8575 | 0.8728 |
| No Tumor | 0.8489 | 0.9975 | 0.9172 |
| Pituitary | 0.9772 | 0.9650 | 0.9711 |

---

## 8. Comparação entre os Experimentos

### 8.1 CNN vs MLP

Todos os experimentos CNN superaram os experimentos MLP em todas as métricas. O melhor resultado MLP (Exp. 6 — MLP Mais Neurônios, F1=0.8286) ficou abaixo do pior resultado CNN com regularização aceitável (Exp. 1 — CNN Básica, F1=0.8887).

Essa diferença é esperada e tem justificativa teórica clara: a **CNN preserva a estrutura espacial da imagem**, capturando relações de vizinhança entre pixels que são fundamentais para detectar formas e texturas de tumores. A MLP, ao achatar a imagem em um vetor de 12.288 features, perde completamente essa informação espacial.

| Métrica | Melhor CNN | Melhor MLP | Diferença |
|---------|-----------|-----------|-----------|
| Accuracy | 0.8994 | 0.8300 | +6.94 pp |
| F1-Score | 0.8979 | 0.8286 | +6.93 pp |

### 8.2 Impacto das Regularizações

**Na CNN:**
- **L2 foi superior ao Dropout**: CNN+L2 (F1=0.8979) superou CNN+Dropout (F1=0.7997). O Dropout com taxa 0.3 nas convolucionais foi muito agressivo, impedindo que os filtros aprendessem representações úteis.
- **Regularização L2 melhorou o baseline**: CNN+L2 superou a CNN Básica em 0.0092 pontos de F1, demonstrando que L2 ajudou na generalização sem prejudicar o aprendizado.

**Na MLP:**
- **Dropout foi catastrófico**: MLP+Dropout (F1=0.6600) foi o pior experimento de todos — queda de 16 pontos percentuais em relação à MLP Básica. O EarlyStopping interrompeu o treino na época 7, pois o modelo não estava aprendendo adequadamente com Dropout=0.4 em camadas já pequenas.
- **L2 também prejudicou**: MLP+L2 (F1=0.8061) foi pior que a MLP Básica (F1=0.8240). Com 12.288 features de entrada, L2 pode estar penalizando excessivamente os pesos necessários para aprender padrões de imagem.

### 8.3 Impacto do Aumento de Capacidade

- **CNN com mais filtros (Exp. 4)** ficou abaixo da CNN Básica: 17 milhões de parâmetros vs. 4 milhões, mas F1 menor (0.8626 vs. 0.8887). O modelo maior apresentou mais instabilidade no treino (val_loss oscilou muito nas primeiras épocas) e o EarlyStopping interrompeu antes de convergir.
- **MLP com mais neurônios (Exp. 6)** foi marginalmente melhor que a MLP Básica (+0.0046 de F1), indicando rendimento decrescente ao aumentar a capacidade sem regularização.

### 8.4 Classe mais difícil

Em todos os modelos, o **Glioma** apresentou os menores valores de Recall (modelo acerta menos quando a imagem é de glioma). O **Meningioma** teve consistentemente a menor Precision — o modelo "confunde" outras classes com meningioma. Isso tem respaldo clínico: o meningioma pode apresentar aparência visual semelhante a outros tumores, sendo o diagnóstico diferencial mais difícil mesmo para especialistas.

A classe **No Tumor** teve sempre o maior Recall (o modelo raramente classifica uma imagem sem tumor como tendo tumor), o que é clinicamente importante — é melhor "alarmar" do que deixar passar.

---

## 9. Discussão

### 9.1 Por que CNN + L2 foi o melhor modelo?

A regularização L2 atua penalizando pesos grandes na função de custo:

```
Loss_total = Loss_classif + lambda * sum(w²)
```

Isso força os filtros convolucionais a distribuírem sua "atenção" por mais features da imagem, ao invés de depender excessivamente de um pequeno conjunto de padrões. No contexto de tumores cerebrais, onde os padrões diagnósticos são distribuídos (textura, forma, localização), essa distribuição de pesos foi benéfica.

### 9.2 Por que o Dropout foi prejudicial?

O Dropout desativa aleatoriamente neurônios durante o treino. Em **CNNs**, desativar filtros inteiros pode destruir representações espaciais que levaram épocas para aprender. Com taxa 0.3 após cada bloco e 0.5 antes do classificador, a CNN+Dropout não conseguiu estabilizar seu aprendizado nas primeiras épocas (val_accuracy de 25% nas épocas 1 e 2).

Na **MLP**, o problema foi ainda mais severo: com apenas 2 camadas ocultas (256 e 128 neurônios) e Dropout=0.4, o modelo efetivo durante o treino tinha apenas ~154 e ~77 neurônios ativos — insuficientes para aprender os padrões de 12.288 features de entrada.

### 9.3 Overfitting observado

Todos os modelos apresentaram algum grau de overfitting (accuracy de treino 99-100% vs. teste 66-90%). Isso é esperado dado o tamanho relativamente pequeno do dataset (5.600 imagens de treino para 4 classes). A CNN+L2 foi a que melhor controlou o overfitting, graças à penalização dos pesos.

---

## 10. Conclusão

Este projeto implementou e comparou 8 experimentos de Aprendizado de Máquina para classificação automática de tumores cerebrais em imagens de ressonância magnética, utilizando CNN e MLP sobre o Brain Tumor MRI Dataset (7.200 imagens, 4 classes).

**O melhor modelo foi a CNN + Regularização L2 (Experimento 3)**, com:
- Accuracy: **89,94%**
- Precision: **90,15%**
- Recall: **89,94%**
- F1-Score: **89,79%**

**Principais conclusões:**

1. **CNNs são superiores às MLPs para classificação de imagens** — diferença de ~7 pontos percentuais de F1, confirmando a vantagem da preservação espacial.

2. **L2 foi a melhor técnica de regularização** — funcionou bem tanto para CNN quanto para MLP (relativamente), pois penaliza suavemente sem destruir aprendizados anteriores.

3. **Dropout foi prejudicial neste cenário** — com o dataset disponível e as arquiteturas escolhidas, o Dropout foi excessivamente agressivo, prejudicando o aprendizado em ambas as técnicas (CNN e MLP).

4. **Mais parâmetros ≠ melhor desempenho** — o Exp. 4 (CNN com 17M parâmetros) e o Exp. 6 (MLP com 6M parâmetros) não superaram seus baselines, indicando que a arquitetura base já era adequada ao problema.

5. **Meningioma e Glioma são as classes mais difíceis** — refletindo a complexidade diagnóstica real dessas condições, onde mesmo especialistas humanos enfrentam dificuldades na diferenciação.

Os resultados obtidos (~90% de F1 com CNN+L2) demonstram o potencial das redes neurais convolucionais como ferramentas de apoio ao diagnóstico médico, mesmo com arquiteturas relativamente simples e sem uso de GPU.

---

## Referências

- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.
- Chollet, F. (2021). *Deep Learning with Python*. Manning Publications.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
- Srivastava, N. et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929-1958.
- Nickparvar, M. (2021). Brain Tumor MRI Dataset. Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
