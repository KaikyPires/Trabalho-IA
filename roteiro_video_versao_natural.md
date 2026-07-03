
# Roteiro de Apresentação (Versão mais natural)

## Parte 1 – Slides

### Slide 1
Olá, professor! Tudo bem?

Neste vídeo vou apresentar o projeto que desenvolvi para a disciplina. O objetivo do trabalho foi classificar tumores cerebrais a partir de imagens de ressonância magnética usando técnicas de Inteligência Artificial.

Primeiro vou explicar a proposta e como os experimentos foram planejados. Depois vou abrir o código para mostrar como tudo foi implementado. No final, volto para os slides para comentar os resultados e as conclusões.

---

### Slide 2

Começando pelo conjunto de dados, utilizei um dataset com 7.200 imagens de ressonância magnética.

Um ponto importante é que esse conjunto de dados é balanceado. São 1.800 imagens para cada uma das quatro classes: glioma, meningioma, pituitary e a classe "No Tumor", que representa imagens sem tumor.

Esse equilíbrio ajuda bastante no treinamento porque evita que o modelo aprenda mais uma classe do que outra.

---

### Slide 3

Para resolver o problema, decidi comparar duas arquiteturas diferentes: uma CNN e uma MLP.

A CNN foi criada para trabalhar com imagens. Ela consegue identificar padrões como bordas, texturas e formatos, mantendo a posição dos pixels durante o processamento.

Já a MLP funciona de outra forma. Antes de treinar, a imagem precisa ser transformada em um vetor, então ela perde essa informação espacial. Justamente por isso ela foi usada como comparação, para ver o quanto essa diferença influencia no resultado final.

---

### Slide 4

Depois defini oito experimentos para comparar diferentes configurações.

Os experimentos 1 e 5 são os modelos básicos. A partir deles fui adicionando Dropout e regularização L2 para observar como cada arquitetura reagia.

O Dropout desativa alguns neurônios durante o treinamento para evitar que a rede fique dependente de um único caminho de aprendizado.

Já a regularização L2 reduz pesos muito altos, ajudando o modelo a generalizar melhor.

Agora que a ideia do projeto foi apresentada, vou abrir o código para mostrar como isso foi implementado.

---

## Parte 2 – Código

### Organização

Aqui está a estrutura do projeto.

Procurei organizar tudo em módulos para deixar o código mais fácil de entender e de manter.

Começando pelo `data_loader.py`, é aqui que as imagens são preparadas antes do treinamento.

Para a CNN mantive imagens de 128 por 128 pixels. Já para a MLP reduzi para 64 por 64, porque ela transforma toda a imagem em um vetor. Se eu mantivesse 128 por 128, a quantidade de entradas seria muito grande e o treinamento ficaria bem mais lento.

Também é aqui que acontece a normalização das imagens. Cada pixel é dividido por 255 para que os valores fiquem entre 0 e 1, o que facilita o treinamento.

Depois faço a separação entre treino, validação e teste usando o `train_test_split`. Também utilizei o parâmetro `stratify` para manter a mesma proporção de classes em cada conjunto.

---

### experiments.py

Neste arquivo estão todas as configurações dos experimentos.

Em vez de alterar o código a cada teste, deixei tudo centralizado. Assim, cada experimento tem sua própria configuração, como taxa de aprendizado, quantidade de filtros e uso de Dropout ou L2.

Isso tornou os testes bem mais organizados.

---

### train_cnn.py

Depois disso entra o `train_cnn.py`.

Ele lê as configurações de cada experimento e monta a rede automaticamente.

Uma parte importante são os callbacks.

O `EarlyStopping` interrompe o treinamento quando percebe que o modelo parou de melhorar. Assim evitamos treinar mais do que o necessário e diminuímos o risco de overfitting.

Também utilizei o `ReduceLROnPlateau`, que reduz automaticamente a taxa de aprendizado quando o treinamento estabiliza. Isso ajuda o modelo a fazer ajustes mais finos.

Como todos os experimentos foram executados em CPU, o treinamento levou algumas horas. Mesmo assim, esses callbacks ajudaram bastante a reduzir o tempo total.

---

### Avaliação

Depois que o treinamento termina, os modelos são carregados pelo `evaluate.py`.

A avaliação é feita usando apenas imagens que o modelo nunca viu antes.

As métricas de acurácia, precisão, recall e F1-Score foram calculadas utilizando funções do Scikit-Learn.

Com essas métricas prontas, foi possível comparar todos os experimentos de forma objetiva.

---

## Parte 3 – Resultados

### Slide 5

Depois de avaliar todos os modelos, chegamos aos resultados.

O melhor desempenho foi do Experimento 3, que utiliza CNN com regularização L2. Ele alcançou aproximadamente 89,9% de acurácia e também apresentou o melhor F1-Score.

Já o pior resultado foi da MLP com Dropout, mostrando que essa combinação não funcionou bem para esse problema.

---

### Slide 6

Analisando os resultados, ficou claro que a CNN aproveitou melhor as informações das imagens.

A regularização L2 conseguiu controlar o overfitting sem prejudicar o aprendizado.

Já o Dropout teve um efeito negativo principalmente na MLP, reduzindo demais a capacidade da rede aprender padrões importantes.

Outro ponto interessante é que uma CNN maior não trouxe melhores resultados. Isso mostra que aumentar a quantidade de parâmetros nem sempre significa melhorar o desempenho.

---

### Slide 7

Para finalizar, os resultados mostraram três pontos principais.

Primeiro, as CNNs tiveram um desempenho melhor do que as MLPs para classificação de imagens.

Segundo, a regularização L2 foi a estratégia que apresentou os melhores resultados.

E, por fim, foi possível observar que um modelo bem ajustado consegue obter um bom desempenho mesmo sem ter uma arquitetura muito grande.

Esses resultados mostram como a escolha da arquitetura e da estratégia de treinamento faz diferença em problemas de classificação de imagens médicas.

Muito obrigado pela atenção!
