# Predição de Diabetes com AutoML (AutoGluon x H2O AutoML)

Este repositório reúne o código e os artefatos utilizados em um estudo comparativo entre dois *frameworks* de AutoML — **AutoGluon** e **H2O AutoML** — aplicados ao problema de predição de diabetes tipo 2 com o conjunto de dados **Pima Indians Diabetes**.

O foco do projeto é:

- Utilizar os *frameworks* de forma **simples**, próxima ao cenário de um usuário não especialista;
- Comparar:
  - desempenho preditivo (Acurácia, Revocação, F1-score);
  - tempo de treinamento;
  - tempo de predição;
  - tipos e quantidade de modelos gerados;
- Permitir testes **interativos** com dados de um novo paciente via terminal.

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
  - Treina vários modelos de classificação com `TabularPredictor` (métrica principal: F1-score, com *bagging*);
  - Gera um *leaderboard* em CSV com métricas e tempos;
  - Permite inserção interativa dos dados de um novo paciente pelo terminal.

- `h2o_main.py`  
  Script principal usando **H2O AutoML**:
  - Lê o mesmo `diabetes.csv`;
  - Converte para `H2OFrame` e define `Outcome` como fator;
  - Executa um experimento de AutoML com limite de modelos;
  - Gera um *leaderboard* em CSV com tempos e métricas;
  - Permite inserção interativa dos dados de um novo paciente.

- `diabetes.csv`  
  Conjunto de dados **Pima Indians Diabetes**, com atributos clínicos/demográficos e a coluna-alvo `Outcome` (0 = sem diabetes, 1 = com diabetes).

- `requirements.txt`  
  Lista de dependências Python utilizadas nos scripts (pandas, h2o, autogluon.tabular, scikit-learn, etc.).

- `Dockerfile`  
  Arquivo para construção da imagem Docker com todo o ambiente necessário.

- `HOWtoRUNDOCKER.txt`  
  Passo a passo resumido para construir e executar o container Docker.

- `TCC_MAIN.pdf`  
  Documento do Trabalho de Conclusão de Curso, com fundamentação teórica e análise detalhada dos resultados.

---

## ⚠️ Pasta `leaderboards` (obrigatória)

Os scripts **salvam os resultados dos experimentos** (tabelas com métricas e tempos) em arquivos CSV dentro de uma pasta chamada **`leaderboards`** na raiz do projeto.

> **Antes de rodar qualquer script, é necessário criar essa pasta manualmente.**

No terminal, a partir da pasta raiz do repositório:

```bash
mkdir leaderboards

