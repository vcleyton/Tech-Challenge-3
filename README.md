# Tech Challenge - Análise de Atrasos em Voos

Previsão e Análise de Atrasos de Aviação nos EUA usando Machine Learning

![Status](https://img.shields.io/badge/status-completo-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

---

## 📋 Visão Geral

Este projeto aplica técnicas de **Machine Learning supervisionado e não supervisionado** para analisar e prever atrasos de voos nos Estados Unidos usando dados de 2015. O projeto implementa um **pipeline completo de ciência de dados**, abrangendo desde análise exploratória até modelagem preditiva.

### Principais Objetivos

- 🔍 **Análise Exploratória (EDA)** - Entender padrões de atrasos e distribuição dos dados
- 🤖 **Aprendizado Supervisionado** - Prever se um voo sofrerá atraso e estimar duração
- 🎯 **Aprendizado Não Supervisionado** - Clusterização de companhias aéreas e aeroportos
- 📊 **Análise Crítica** - Insights, limitações e recomendações

---

## 📁 Estrutura do Projeto

```
Tech-Challenge-3/
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências do projeto
├── data/                              # Dados brutos
│   ├── airlines.csv                   # Informações sobre companhias aéreas
│   ├── airports.csv                   # Informações sobre aeroportos
│   └── flights.csv                    # Dados de voos com informações de atrasos
└── notebooks/
    └── flight_delays_analysis.ipynb   # Notebook principal com análise completa
```

---

## 📊 Dados Utilizados

### Conjuntos de Dados

- **airlines.csv** - 14 companhias aéreas operando nos EUA em 2015
- **airports.csv** - 395 aeroportos norte-americanos
- **flights.csv** - 5.819.079 registros de voos em 2015

### Características Principais dos Dados

- **Período**: Ano completo de 2015
- **Cobertura**: Todos os voos domésticos dos EUA
- **Threshold de Atraso**: Voos com atraso de chegada > 15 minutos classificados como "atrasados"
- **Features**: Horários programados/reais, Origem/Destino, Companhia Aérea, Motivos de atraso, etc.

### Fonte dos Dados

**US Flight Delays and Cancellations (2015)** - Dataset público com 5.819.079 registros de voos

#### Download dos Dados

Os dados utilizados neste projeto podem ser obtidos através do link:

🔗 [Google Drive - Tech Challenge 3 Data](https://drive.google.com/drive/folders/1aS7exW5N0qq1uIxvIBcAfc18OHojOMjj)

Certifique-se de fazer o download dos arquivos (`airlines.csv`, `airports.csv` e `flights.csv`) e coloque-os na pasta `data/` antes de executar o notebook.

---

## 🛠️ Requisitos e Instalação

### Pré-requisitos

- **Python 3.8+**
- **pip** ou **conda** para gerenciamento de pacotes

### Dependências Principais

| Pacote | Propósito |
|--------|-----------|
| `pandas` | Manipulação e análise de dados |
| `numpy` | Computação numérica |
| `scikit-learn` | Algoritmos de Machine Learning |
| `matplotlib` | Visualizações estáticas |
| `seaborn` | Visualizações estatísticas |
| `plotly` | Visualizações interativas |
| `xgboost` | Gradient Boosting avançado |
| `scipy` | Computação científica |
| `statsmodels` | Modelagem estatística |
| `imbalanced-learn` | Técnicas de balanceamento |

> **Nota**: Alguns pacotes são opcionais (plotly, xgboost, statsmodels) mas recomendados para análise completa.

### Configuração do Ambiente

#### Opção 1: Usando venv (Recomendado)

```bash
# Clonar ou navegar até o diretório do projeto
cd Tech-Challenge-3

# Criar ambiente virtual
python -m venv venv

# Ativar ambiente (Windows)
venv\Scripts\activate

# Ativar ambiente (macOS/Linux)
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

#### Opção 2: Usando Conda

```bash
conda create -n flight-delays python=3.8
conda activate flight-delays
pip install -r requirements.txt
```

---

## 🚀 Como Executar

### Executar o Notebook Jupyter

```bash
# Certifique-se de que o ambiente virtual está ativado
jupyter notebook

# ou use JupyterLab (recomendado)
jupyter lab
```

Então, navegue até `notebooks/flight_delays_analysis.ipynb` e abra o notebook.

---

## 📚 Conteúdo do Notebook

O notebook está organizado em **7 seções principais**:

### 1️⃣ **Data Loading and Preparation**
- Importação de bibliotecas necessárias
- Carregamento dos três datasets (flights, airports, airlines)
- Verificação inicial de estrutura e tipos de dados

### 2️⃣ **Exploratory Data Analysis (EDA)**
- Inspeção detalhada de dados faltantes
- Estatísticas descritivas
- Distribuição de atrasos por:
  - Companhia aérea (14 airlines)
  - Aeroporto (395 airports)
  - Dia da semana / Mês / Hora do dia
  - Motivos de atraso
- Análise de correlações

### 3️⃣ **Data Preprocessing and Feature Engineering**
- Tratamento de valores faltantes
- Codificação de variáveis categóricas (LabelEncoder)
- Normalização de features (StandardScaler)
- Engenharia de features (hora do dia, período do dia, distância, etc.)
- Criação de variável alvo (IS_DELAYED para classificação, duração para regressão)

### 4️⃣ **Supervised Learning - Classification**
Modelos para prever se um voo sofrerá atraso (>15 minutos):

- **Random Forest Classifier** (MELHOR MODELO)
  - Accuracy: 66%
  - Precision: 29% (quando prevê atraso, acerta 29% das vezes)
  - Recall: 64% (detecta 64% dos voos realmente atrasados)
  - ROC-AUC: 0.71 (excelente discriminação)

- **Logistic Regression**
  - Baseline com performance linear

- **Decision Tree**
  - Performance moderada

### 5️⃣ **Supervised Learning - Regression**
Modelos para prever a duração exata do atraso em minutos:

- **Gradient Boosting Regressor** (MELHOR MODELO) ⭐
  - MAE: ~21 minutos (erro médio)
  - RMSE: ~39 minutos

- **Random Forest Regressor**
  - Performance similar ao GB

- **Linear Regression**
  - Performance fraca

### 6️⃣ **Unsupervised Learning - Clustering**
- **K-Means Clustering** para agrupar aeroportos
- Determinação do número ótimo de clusters (Silhueta + Elbow Method)
- **PCA (Principal Component Analysis)** para visualização em 2D
- Análise de centróides e características dos clusters

### 7️⃣ **Critical Analysis of Results**
Análise abrangente incluindo:
- Resumo executivo do projeto
- Métricas consolidadas de performance
- Importância das features (quais fatores mais influenciam atrasos)
- Principais conclusões
- Limitações dos modelos
- Casos de uso recomendados
- Propostas de melhorias (curto, médio e longo prazo)

---

## 🔧 Tecnologias Utilizadas

### Linguagem e Ambiente
- **Python 3.8+**
- **Jupyter Notebook**

### Bibliotecas de Dados
- Pandas - Manipulação de dados
- NumPy - Computação numérica
- SciPy - Estatísticas científicas

### Machine Learning
- scikit-learn - Algoritmos ML
- XGBoost - Gradient Boosting
- imbalanced-learn - Balanceamento

### Visualização
- Matplotlib - Gráficos estáticos
- Seaborn - Visualizações estatísticas
- Plotly - Gráficos interativos

---

## ⚠️ Limitações e Recomendações

### Principais Limitações dos Modelos

**Dados Faltantes:**
- ❌ Dados de clima (grande impacto em atrasos)
- ❌ Informações de manutenção de aeronaves
- ❌ Dados de congestionamento ATC
- ❌ Detalhes de operações em terra
- ❌ Histórico de voos anteriores na cadeia

**Escopo Temporal:**
- ⏰ Apenas dados de 2015 (pode não generalizar para outros anos)
- 🌍 Padrões pós-COVID não capturados
- 📊 Mudanças operacionais das companhias aéreas não refletidas

**Performance do Modelo:**
- 📉 77% da variância permanece inexplicada
- 🎯 Pior performance para atrasos extremos
- 🔄 Sem mecanismo de atualização com novos dados

### Recomendações Curto Prazo

1. **Engenharia de Features Avançada**
   - Criar features de interação (airline × airport)
   - Adicionar features de congestionamento cumulativo
   - Incluir histórico de atrasos anteriores
   - **Impacto esperado**: +3-5% em accuracy/R²

2. **Otimização de Hiperparâmetros**
   - GridSearchCV / RandomSearchCV
   - Cross-validation estratificada
   - Balanceamento de classes (SMOTE)
   - **Impacto esperado**: +2-4% em performance

3. **Análise de Interpretabilidade**
   - SHAP values para explicações locais
   - Partial dependence plots
   - Mensagens de confiança para predições

### Recomendações Médio Prazo

1. **Dados Adicionais**
   - Integração de dados de clima (weatherAPI)
   - Métricas de congestionamento ATC
   - Dados operacionais das companhias aéreas
   - **Impacto esperado**: +10-15% em accuracy

2. **Modelos Temporais**
   - Séries temporais (ARIMA, LSTM, Prophet)
   - Melhor captura de sazonalidade
   - Tratamento de tendências
   - **Impacto esperado**: +5-10% para previsões curto prazo

### Recomendações Longo Prazo

1. **Sistema de Produção**
   - Pipeline de streaming para dados em tempo real
   - Atualização automática de modelos
   - Integração com sistemas operacionais das companhias

2. **Análise Causal**
   - Identificar causas raiz (não apenas correlações)
   - Modelagem de impacto de mudanças operacionais
   - Análise contrafactual

3. **Modelos Especializados**
   - Modelos separados por rota/companhia/período
   - **Impacto esperado**: +15-25% em acurácia específica

---

## 🎯 Casos de Uso Recomendados

✅ **RECOMENDADO:**
- Planejamento estratégico de recursos
- Análise retrospectiva de padrões
- Pricing e gestão de receita
- Comunicações proativas com clientes

⚠️ **CONDICIONADO:**
- Rebooking automático (validar threshold de confiança)
- Ajuste de horários (validar com especialistas)

❌ **NÃO RECOMENDADO:**
- Decisões operacionais em tempo real (< 30 min antes)
- Conformidade regulatória (modelo precisa ser mais transparente)
- Garantias ao cliente individual

---

## 🎯 Como Usar Este Projeto

1. **Clone ou baixe** o repositório
2. **Configure o ambiente** usando os passos na seção "Requisitos e Instalação"
3. **Instale as dependências** via `pip install -r requirements.txt`
4. **Abra o notebook** em Jupyter ou JupyterLab
5. **Execute as células sequencialmente** para reproduzir a análise
6. **Explore os resultados** - gráficos, métricas e insights estarão disponíveis

---

## 📝 Notas Importantes

- ⚠️ **Dados Históricos**: Dataset de 2015 - padrões podem ter mudado significativamente
- 🔄 **Notebook Autocontido**: Pode ser executado sequencialmente do início ao fim
- 💾 **Requisitos de Memória**: ~2-4GB para processar dataset completo
- 📦 **Dependências Opcionais**: xgboost, plotly, statsmodels são opcionais mas recomendados
- 🎯 **Threshold de Atraso**: Definido como 15 minutos (pode ser ajustado conforme necessidade)
- 🔍 **Reproducibilidade**: Seeds aleatórias definidas para resultados consistentes

---

**Última atualização**: Maio 2026
