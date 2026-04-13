# BreastCare AI — Classificação de Tumores de Mama

> MVP desenvolvido para a pós-graduação em Engenharia de Software da **PUC-Rio**.  
> Sistema completo de classificação de tumores mamários (maligno/benigno) a partir de medições morfométricas de biópsias por aspiração com agulha fina (FNA).

---

## Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Arquitetura](#arquitetura)
- [Tecnologias](#tecnologias)
- [Estrutura de Pastas](#estrutura-de-pastas)
- [Machine Learning](#machine-learning)
- [API](#api)
- [Frontend](#frontend)
- [Testes](#testes)
- [Como Executar](#como-executar)
- [Considerações de Segurança](#considerações-de-segurança)

---

## Sobre o Projeto

O **BreastCare AI** é uma ferramenta de suporte diagnóstico que utiliza aprendizado de máquina para classificar tumores mamários como **malignos** ou **benignos** com base em 10 features morfométricas extraídas de imagens digitalizadas de biópsias.

O dataset utilizado é o **Breast Cancer Wisconsin (Diagnostic)**, disponível via `scikit-learn`, com 569 amostras e duas classes:

| Classe | Descrição | Amostras |
|--------|-----------|----------|
| 0 | Maligno | 212 |
| 1 | Benigno | 357 |

> **Aviso:** Esta ferramenta é destinada exclusivamente a fins educacionais e de pesquisa. Não substitui avaliação clínica, exame histopatológico ou julgamento médico especializado.

---

## Arquitetura

```
Frontend (HTML/CSS/JS)
        │
        │  POST /prediction  (JSON)
        ▼
API REST (FastAPI + Uvicorn)
        │
        ├── Validação de entrada (Pydantic)
        ├── Pré-processamento (StandardScaler)
        ├── Inferência (KNN treinado)
        └── Persistência (SQLite via SQLAlchemy)
                │
                ▼
        ml/melhor_modelo.pkl
        ml/scaler.pkl
        ml/colunas.pkl
```

---

## Tecnologias

| Camada | Tecnologia | Versão |
|--------|-----------|--------|
| Linguagem | Python | ≥ 3.12 |
| API | FastAPI | ≥ 0.135 |
| Servidor | Uvicorn | ≥ 0.44 |
| ML | scikit-learn | ≥ 1.8 |
| ORM | SQLAlchemy | ≥ 2.0 |
| Banco de dados | SQLite | — |
| Validação | Pydantic v2 | — |
| Testes | pytest + httpx | ≥ 9.0 |
| Frontend | Bootstrap 5 + Vanilla JS | 5.3.3 |
| Gerenciador de pacotes | uv | — |

---

## Estrutura de Pastas

```
breast_cancer_classification-main/
│
├── app/                          # Aplicação FastAPI
│   ├── main.py                   # Inicialização da app e middlewares
│   ├── database.py               # Configuração do SQLAlchemy (SQLite)
│   ├── models/
│   │   ├── samples.py            # Modelo ORM — tabela samples
│   │   └── predictions.py        # Modelo ORM — tabela predictions
│   ├── routers/
│   │   ├── __init__.py           # Registro dos roteadores
│   │   └── prediction.py         # Endpoint POST /prediction
│   ├── schemas/
│   │   └── sample.py             # Schema Pydantic de entrada
│   └── utils/
│       └── load_classifier.py    # Carregamento dos artefatos .pkl
│
├── ml/                           # Machine Learning
│   ├── breast_cancer_wisconsin_classifier.ipynb  # Notebook de treinamento
│   ├── melhor_modelo.pkl         # Modelo KNN serializado
│   ├── scaler.pkl                # StandardScaler serializado
│   └── colunas.pkl               # Lista de features utilizada
│
├── frontend/                     # Interface web
│   ├── index.html                # Página principal
│   ├── style.css                 # Estilos
│   └── script.js                 # Lógica de submissão e exibição
│
├── tests/                        # Suíte de testes
│   ├── test_api.py               # Testes da API (22 casos)
│   └── test_model.py             # Testes do modelo ML (22 casos)
│
├── instance/
│   └── results.db                # Banco SQLite com histórico de predições
│
├── pyproject.toml                # Dependências e metadados do projeto
└── README.md
```

---

## Machine Learning

O notebook `ml/breast_cancer_wisconsin_classifier.ipynb` documenta todo o pipeline de treinamento.

### Features utilizadas

Apenas as 10 primeiras features `_mean` do dataset Wisconsin:

| Feature | Descrição |
|---------|-----------|
| `radius` | Média das distâncias do centro até os pontos do perímetro |
| `texture` | Desvio padrão dos valores em escala de cinza |
| `perimeter` | Tamanho médio do contorno do núcleo |
| `area` | Área média do tumor |
| `smoothness` | Variação local nos comprimentos do raio |
| `compactness` | Perímetro² / área − 1,0 |
| `concavity` | Severidade das porções côncavas do contorno |
| `concave_points` | Número de porções côncavas do contorno |
| `symmetry` | Simetria do núcleo celular |
| `fractal_dimension` | Aproximação da linha costeira − 1 |

### Metodologia

- **Divisão treino/teste:** 80/20, estratificada por classe (`random_state=42`)
- **Pré-processamento:** `StandardScaler` dentro de um `Pipeline` para evitar vazamento de dados
- **Seleção de modelo:** `GridSearchCV` com `StratifiedKFold` (5 folds) em 4 algoritmos

### Resultados comparativos

| Modelo | CV Accuracy | Test Accuracy | Precision | Recall | F1-Score |
|--------|-------------|---------------|-----------|--------|----------|
| **KNN** ✅ | **0.9429** | **0.9211** | 0.8667 | 0.9286 | **0.8966** |
| SVM | 0.9429 | 0.9211 | 0.8667 | 0.9286 | 0.8966 |
| Naive Bayes | 0.9121 | 0.9211 | 0.8837 | 0.9048 | 0.8941 |
| Árvore de Decisão | 0.9209 | 0.8947 | 0.8000 | 0.9524 | 0.8696 |

> Métricas calculadas sobre a **classe maligna** (pos_label=0), que é a mais crítica clinicamente.

### Modelo selecionado: KNN

```
Melhores hiperparâmetros: n_neighbors=3, weights='uniform'

              precision    recall  f1-score   support
     Maligno       0.87      0.93      0.90        42
     Benigno       0.96      0.92      0.94        72
    accuracy                           0.92       114
```

O modelo foi selecionado pelo maior **F1-Score** sobre a classe maligna, pois em diagnóstico médico o **Recall** é a métrica mais crítica — um falso negativo (tumor maligno classificado como benigno) é clinicamente mais grave do que um falso positivo.

### Artefatos exportados

| Arquivo | Conteúdo |
|---------|----------|
| `melhor_modelo.pkl` | Classificador KNN treinado |
| `scaler.pkl` | StandardScaler ajustado ao conjunto de treino |
| `colunas.pkl` | Lista ordenada das 10 features |

---

## API

### Executar o servidor

```bash
# A partir da raiz do projeto
cd app
python main.py
```

O servidor iniciará em `http://localhost:8000`.

A documentação interativa (Swagger UI) estará disponível em:  
`http://localhost:8000/docs`

### Endpoints

#### `GET /`
Verifica se a API está operacional.

**Resposta:**
```json
{ "message": "Sucess, the API is working" }
```

---

#### `POST /prediction`
Recebe as medições morfométricas e retorna a classificação.

**Corpo da requisição (`application/json`):**

```json
{
  "radius": 14.12,
  "texture": 19.29,
  "perimeter": 91.97,
  "area": 654.8,
  "smoothness": 0.096,
  "compactness": 0.104,
  "concavity": 0.088,
  "concave_points": 0.048,
  "symmetry": 0.181,
  "fractal_dimension": 0.062
}
```

**Resposta (`200 OK`):**

```json
{
  "result": "Benign",
  "confiability": 94.23,
  "probability_malignant": 5.77,
  "probability_benign": 94.23,
  "alert": false
}
```

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `result` | string | `"Malignant"` ou `"Benign"` |
| `confiability` | float | Confiança da predição (0–100) |
| `probability_malignant` | float | Probabilidade da classe maligna (%) |
| `probability_benign` | float | Probabilidade da classe benigna (%) |
| `alert` | bool | `true` se o resultado for maligno |

**Erros possíveis:**

| Código | Motivo |
|--------|--------|
| `422` | Campo ausente ou tipo inválido |
| `503` | Arquivos `.pkl` não encontrados |

### Persistência

Cada predição é salva automaticamente no banco SQLite (`instance/results.db`) em duas tabelas:

- `samples` — armazena os valores morfométricos recebidos
- `predictions` — armazena o resultado, confiabilidade e referência à amostra

---

## Frontend

Interface web construída com **Bootstrap 5**, JavaScript puro e a paleta de cores `#005354 / #BBFDFD / #ADEEEF / #91D2D2`.

### Executar o frontend

Abra o arquivo `frontend/index.html` via um servidor HTTP local (não via `file://`, pois o browser bloqueia requisições `fetch` para `http://` a partir de `file://`):

```bash
# Python
cd frontend
python -m http.server 5500
```

Acesse `http://localhost:5500` no navegador. A API deve estar rodando em paralelo na porta `8000`.

### Funcionalidades

- Formulário com os 10 campos morfométricos, organizados em 3 grupos semânticos
- Valores padrão pré-preenchidos (amostra representativa do dataset)
- Botão de reset para os valores padrão
- Exibição do diagnóstico, confiança e barras de probabilidade animadas
- Timeout de 10 segundos com mensagem de erro descritiva
- Aviso clínico de uso como ferramenta de suporte apenas

---

## Testes

A suíte cobre **44 casos de teste** divididos em dois arquivos.

### Executar os testes

```bash
# A partir da raiz do projeto
pytest tests/ -v
```

### `tests/test_api.py` — 22 testes

| Classe | O que testa |
|--------|-------------|
| `TestHealthCheck` | Endpoint raiz retorna 200 e mensagem |
| `TestPredictionHappyPath` | Campos da resposta, soma das probabilidades, lógica do campo `alert`, confiabilidade |
| `TestPredictionValidation` | Campos faltando, tipos errados, corpo vazio retornam 422 |
| `TestPredictionContentType` | Rejeita form-encoded, responde em JSON |
| `TestPredictionDeterminism` | Mesma entrada sempre produz mesma saída |

### `tests/test_model.py` — 22 testes

| Classe | O que testa |
|--------|-------------|
| `TestArtefacts` | Existência e integridade dos 3 arquivos `.pkl` |
| `TestScaler` | Shape de saída, valores finitos, escalonamento de amostra única |
| `TestPredictionOutput` | Shapes, labels válidos, probabilidades somam 1 |
| `TestModelPerformance` | Accuracy ≥ 90%, Recall maligno ≥ 90%, F1 ≥ 88%, determinismo |

---

## Como Executar

### Pré-requisitos

- Python ≥ 3.12
- `uv` (gerenciador de pacotes) **ou** `pip`

### Instalação

```bash
# Clone o repositório
git clone <url-do-repositório>
cd breast_cancer_classification-main

# Com uv (recomendado)
uv sync

# Ou com pip
pip install -r requirements.txt
```

### Passo a passo

```bash
# 1. Iniciar a API
cd app
python main.py
# → API rodando em http://localhost:8000

# 2. Em outro terminal, servir o frontend
cd frontend
python -m http.server 5500
# → Frontend em http://localhost:5500

# 3. Em outro terminal, rodar os testes
cd ..
pytest tests/ -v
```

---

## Considerações de Segurança

- **Anonimização:** dados de saúde devem ser anonimizados antes do treinamento — o dataset Wisconsin já é público e não contém identificadores pessoais.
- **Minimização de features:** apenas as 10 features `_mean` foram utilizadas, reduzindo a superfície de exposição.
- **Trânsito:** em produção, a API deve ser servida exclusivamente via **HTTPS**.
- **LGPD:** o histórico de predições armazenado no SQLite garante rastreabilidade do uso do modelo, requisito essencial para conformidade com a Lei Geral de Proteção de Dados.
- **Uso responsável:** o campo `alert` e o aviso clínico no frontend reforçam que o sistema é uma ferramenta de **suporte à decisão**, não um substituto do diagnóstico médico.

---

<div align="center">
  <sub>Desenvolvido como MVP da Pós-Graduação em Engenharia de Software — PUC-Rio</sub>
</div>
