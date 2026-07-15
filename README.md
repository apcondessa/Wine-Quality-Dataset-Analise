# 🍷 Classificação da Qualidade de Vinhos com Machine Learning

> **Tech Challenge — FIAP | Pós-Tech em Data Analytics**  
> Projeto de classificação binária utilizando dados físico-químicos de vinhos para prever se um vinho é de **alta qualidade** ou **baixa/média qualidade**.

---

## 📋 Sumário

- [Contexto](#-contexto)
- [Objetivo](#-objetivo)
- [Dataset](#-dataset)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Metodologia](#-metodologia)
- [Modelos Treinados](#-modelos-treinados)
- [Métricas de Avaliação](#-métricas-de-avaliação)
- [Como Executar](#-como-executar)
- [Resultados Esperados](#-resultados-esperados)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)

---

## 🎯 Contexto

A indústria vitivinícola tradicionalmente utiliza avaliações sensoriais realizadas por especialistas para classificar a qualidade de vinhos. Esse processo, embora importante, é subjetivo, demorado e fortemente dependente da experiência individual dos avaliadores.
Este projeto propõe uma abordagem baseada em dados: utilizar características físico-químicas mensuráveis dos vinhos como entrada para modelos de Machine Learning capazes de classificar automaticamente a qualidade.

---

## 🎯 Objetivo

Desenvolver e comparar modelos de classificação binária que, a partir de variáveis físico-químicas, prevejam se um vinho possui:

- **Alta qualidade** → nota original `quality >= 7` → classe `1`  
- **Baixa ou média qualidade** → nota original `quality < 7` → classe `0`

---

## 📦 Dataset

**Arquivo:** `WineQT.csv`

A base contém amostras de vinho tinto com as seguintes variáveis:

| Variável | Descrição |
|---|---|
| `fixed acidity` | Acidez fixa |
| `volatile acidity` | Acidez volátil |
| `citric acid` | Ácido cítrico |
| `residual sugar` | Açúcar residual |
| `chlorides` | Cloretos |
| `free sulfur dioxide` | Dióxido de enxofre livre |
| `total sulfur dioxide` | Dióxido de enxofre total |
| `density` | Densidade |
| `pH` | pH |
| `sulphates` | Sulfatos |
| `alcohol` | Teor alcoólico |
| `quality` | Nota de qualidade atribuída por especialistas (variável original) |
| `high_quality` | **Variável alvo criada:** `1` = alta qualidade, `0` = baixa/média |

> A coluna `Id` é removida antes da modelagem por ser apenas um identificador da amostra.

---

## 📁 Estrutura do Projeto

```
wine-quality-ml/
│
├── data/
│   └── WineQT.csv                  # Base de dados original
│
├── models/                         # Modelos treinados salvos (.pkl)
│   ├── regressão_logística.pkl
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
│
├── notebooks/
│   └── wine_quality_classification.ipynb  # Notebook principal com análises e resultados
│
├── src/                            # Scripts auxiliares (pacote Python)
│   ├── __init__.py                 # Expõe o pacote e todos os símbolos públicos
│   ├── data_loader.py              # Carregamento, validação e pré-processamento
│   ├── eda.py                      # Análise exploratória e visualizações
│   ├── train.py                    # Treinamento, validação cruzada e persistência
│   ├── evaluate.py                 # Métricas, matrizes, curvas ROC e importância
│   └── pipeline.py                 # Orquestrador do fluxo completo (CLI)
│
├── requirements.txt                # Dependências do projeto
└── README.md
```

---

## 🔬 Metodologia

O projeto seguiu as etapas clássicas de um pipeline de Machine Learning:

### 1. Carregamento e inspeção dos dados
Importação do `WineQT.csv` com verificação de shape, tipos de dados, valores nulos e estatísticas descritivas.

### 2. Criação da variável alvo binária
A nota original `quality` foi transformada em uma variável binária `high_quality`, simplificando o problema para uma classificação de duas classes.

### 3. Análise exploratória de dados (EDA)
- Distribuição da nota original de qualidade e das classes binárias
- Histogramas das variáveis físico-químicas
- Boxplots por classe de qualidade para identificar variáveis discriminantes
- Matriz de correlação e correlação das features com a variável alvo

### 4. Análise de outliers
Identificação de valores extremos pelo método IQR (Intervalo Interquartil). Os outliers foram mantidos na base por representarem variações legítimas do processo produtivo.

### 5. Separação treino/teste
Divisão 80/20 com estratificação pela variável alvo, garantindo que a proporção entre classes seja preservada em ambos os conjuntos.

### 6. Pré-processamento
Padronização via `StandardScaler` aplicada dentro de um `Pipeline` do scikit-learn para evitar vazamento de dados (*data leakage*).

### 7. Treinamento dos modelos
Três algoritmos foram treinados e comparados (detalhes abaixo).

### 8. Avaliação
Comparação com múltiplas métricas, análise de matrizes de confusão, curvas ROC e importância de variáveis.

---

## 🤖 Modelos Treinados

### Regressão Logística
Modelo base, simples e interpretável. Aplicado com `StandardScaler` via Pipeline e balanceamento de classes (`class_weight="balanced"`).

### Random Forest
Ensemble baseado em árvores de decisão. Captura relações não lineares entre as variáveis e é menos sensível a outliers. Configurado com 300 estimadores e balanceamento de classes.

### Gradient Boosting
Aprendizado sequencial e robusto. Treinado com 200 estimadores e taxa de aprendizado de 0.05 para reduzir overfitting.

---

## 📊 Métricas de Avaliação

Como a base possui **desbalanceamento de classes** (a maioria dos vinhos é de baixa/média qualidade), a acurácia isolada pode ser enganosa. Por isso, foram utilizadas as seguintes métricas:

| Métrica | Descrição |
|---|---|
| **Acurácia** | Percentual geral de acertos |
| **Precision** | Entre os vinhos previstos como alta qualidade, quantos realmente eram |
| **Recall** | Entre os vinhos de alta qualidade, quantos o modelo identificou corretamente |
| **F1-score** | Média harmônica entre Precision e Recall |
| **ROC-AUC** | Capacidade do modelo de separar as duas classes |

> O **F1-score** e o **Recall da classe positiva** são as métricas prioritárias, pois identificar corretamente os vinhos de alta qualidade é o objetivo principal do problema.

---

## ▶️ Como Executar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Opção 1 — Notebook (Google Colab / Jupyter)

Abra o arquivo `notebooks/wine_quality_classification.ipynb` e execute as células em sequência.

### Opção 2 — Pipeline via linha de comando

```bash
# Executar o pipeline completo (com EDA)
python src/pipeline.py --data_path data/WineQT.csv

# Executar sem os gráficos exploratórios
python src/pipeline.py --data_path data/WineQT.csv --skip_eda

# Executar sem salvar os modelos em disco
python src/pipeline.py --data_path data/WineQT.csv --no_save
```

### Opção 3 — Importar módulos individualmente

```python
import sys
sys.path.append("src")

from data_loader import load_data, create_binary_target, split_features_target
from train import build_models, train_models, split_data
from evaluate import evaluate_all_models, plot_roc_curves, plot_feature_importance

# Carregar e preparar os dados
df = load_data("data/WineQT.csv")
df = create_binary_target(df)
X, y = split_features_target(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Treinar e avaliar
models = build_models()
trained = train_models(models, X_train, y_train)
results = evaluate_all_models(trained, X_test, y_test)
print(results)
```

---

## 📈 Resultados Esperados

Os três modelos são comparados em uma tabela consolidada de métricas. De forma geral, espera-se que:

- **Gradient Boosting** e **Random Forest** superem a Regressão Logística em F1-score e ROC-AUC
- As variáveis mais relevantes para a previsão sejam **teor alcoólico**, **acidez volátil** e **sulfatos**
- O modelo escolhido apresente bom equilíbrio entre Precision e Recall na classe de alta qualidade

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| `pandas` | 2.2.2 | Manipulação de dados |
| `numpy` | 1.26.4 | Operações numéricas |
| `matplotlib` | 3.9.2 | Visualizações gráficas |
| `seaborn` | 0.13.2 | Gráficos estatísticos |
| `scikit-learn` | 1.5.2 | Modelos, métricas e pré-processamento |

---

## 👤 Autor

Desenvolvido como entrega do **Tech Challenge — Desafio 2** da Pós-Tech em Data Analytics da **FIAP**.
