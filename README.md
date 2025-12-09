# Predição de Diabetes com AutoML (AutoGluon x H2O AutoML)

Este repositório reúne o código e os artefatos utilizados em um estudo comparativo entre dois *frameworks* de AutoML — **AutoGluon** e **H2O AutoML** — aplicados ao problema de predição de diabetes tipo 2 com o conjunto de dados **Pima Indians Diabetes**.

O foco do projeto é:

- Utilizar os *frameworks* de forma **simples**, próxima ao cenário de um usuário não especialista;
- Comparar:
  - desempenho preditivo (Acurácia, Revocação, F1-score);
  - tempo de treinamento;
  - tempo de predição;
  - tipos e quantidade de modelos gerados (árvores, *gradient boosting*, redes neurais, GLM, etc.);
- Permitir testes **interativos** com dados de um novo paciente direto pelo terminal.

> ⚠️ **Aviso importante**  
> Os modelos são treinados e avaliados essencialmente **no mesmo conjunto de dados**, usando apenas validações internas.  
> Isso aumenta o risco de **sobreajuste (overfitting)** e de métricas artificialmente elevadas.  
> Este projeto tem finalidade **acadêmica e exploratória** e **não deve ser utilizado** para suporte real a diagnóstico ou decisões clínicas.

---

## Estrutura do Repositório

Principais arquivos e pastas:

- `autogluon_main.py`  
  Script principal usando **AutoGluon Tabular**:
  - Lê o arquivo `diabetes.csv`;
  - Treina vários modelos de classificação com `TabularPredictor` (métrica principal: F1-score, com *bagging* `num_bag_folds=10`); :contentReference[oaicite:0]{index=0}  
  - Gera um *leaderboard* consolidado com:
    - nome do modelo;
    - tempo de treino;
    - tempo de predição;
    - Acurácia, Revocação e F1-score calculadas modelo a modelo;
  - Permite informar **manual e interativamente** os dados de um novo paciente pelo terminal
    e retorna:
    - classe prevista (0 = NÃO tem diabetes, 1 = TEM diabetes);
    - probabilidade da classe 1.

- `h2o_main.py`  
  Script principal usando **H2O AutoML**:
  - Lê o mesmo `diabetes.csv`;
  - Converte o *DataFrame* para `H2OFrame` e define a coluna `Outcome` como fator (classificação binária); :contentReference[oaicite:1]{index=1}  
  - Executa um experimento de AutoML com:
    - `nfolds=10`;
    - `max_models=30`;
    - `seed=42`;
  - Monta um *leaderboard* em CSV com:
    - `model_id`;
    - `training_time_ms`;
    - `predict_time_per_row_ms`;
    - Acurácia, Revocação e F1-score para cada modelo (no mesmo conjunto de dados usado no treino);
  - Fornece um modo interativo para inserir os dados de um novo paciente e obter:
    - classe prevista (TEM / NÃO TEM diabetes);
    - probabilidade de diabetes (`p1` = probabilidade de `Outcome = 1`).

- `diabetes.csv`  
  Conjunto de dados tabular **Pima Indians Diabetes**, contendo atributos como:
  - `Pregnancies`
  - `Glucose`
  - `BloodPressure`
  - `SkinThickness`
  - `Insulin`
  - `BMI`
  - `DiabetesPedigreeFunction`
  - `Age`
  - `Outcome` (0 = sem diabetes, 1 = com diabetes)

- `requirements.txt`  
  Lista de dependências Python utilizadas tanto nos scripts quanto na imagem Docker (pandas, h2o, autogluon.tabular com LightGBM, CatBoost, FastAI, XGBoost, PyTorch, scikit-learn, etc.). :contentReference[oaicite:2]{index=2}  

- `Dockerfile`  
  Arquivo para construção da imagem Docker, que encapsula todo o ambiente necessário (Python + bibliotecas + scripts).

- `HOWtoRUNDOCKER.txt`  
  Arquivo com os comandos básicos para construir a imagem e executar o container, montando o diretório do projeto no container. :contentReference[oaicite:3]{index=3}  

- `leaderboards/`  
  Pasta onde são salvos os resultados dos experimentos:
  - `autogluon_leaderboard.csv`
  - `h2o_leaderboard.csv`

- `TCC_MAIN.pdf`  
  Versão em PDF do Trabalho de Conclusão de Curso, contendo:
  - contextualização clínica (diabetes mellitus);
  - fundamentação teórica de ML e AutoML;
  - descrição detalhada da metodologia;
  - análise dos resultados gerados por estes scripts.

---

## Ambiente de Execução

Você pode rodar o projeto de duas formas:

1. Usando **Docker** (recomendado, pois isola o ambiente);
2. Usando Python localmente com `requirements.txt`.

### 1. Executando com Docker

#### 1.1. Pré-requisitos

- Docker instalado na máquina (Windows, Linux ou macOS);
- Este repositório clonado em uma pasta local.

#### 1.2. Construir a imagem

Na pasta raiz do repositório (onde está o `Dockerfile`), execute:

```bash
docker build --no-cache -t tcc-automl .
