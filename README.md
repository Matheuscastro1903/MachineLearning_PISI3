<div align="center">

<img src="dash/assets/logo.png" alt="ThinkMoney" width="130" />

# 💸 ThinkMoney — Inteligência de Finanças Pessoais

**Da base de dados financeira à decisão: exploração interativa, diagnóstico de vulnerabilidade e segmentação de clientes em um só lugar.**

🚀 **[Acessar o dashboard online](https://louquinhosbr-new-thinkmoney-dashboard.hf.space/)**

`Python 3.10+` · `Dash` · `Plotly` · `scikit-learn`

<a href="#-visão-geral">Visão geral</a> ·
<a href="#️-o-dashboard">Dashboard</a> ·
<a href="#-análises-exploratórias-eda">EDA</a> ·
<a href="#-modelos-de-machine-learning">Modelos</a> ·
<a href="#️-como-executar">Executar</a>

</div>

---

## 📌 Visão geral

O **ThinkMoney** é um projeto de Ciência de Dados que transforma uma base financeira de **20.000 perfis** em uma experiência completa de análise e previsão. O objetivo é responder, com dados, a três perguntas centrais:

1. **O que os dados revelam?** — uma camada robusta de análise exploratória (EDA) sobre renda, despesas, moradia, consumo e poupança.
2. **Quem está em risco?** — um modelo de **classificação** que identifica usuários em situação de **vulnerabilidade financeira**.
3. **Quem são os clientes?** — um modelo de **clusterização** que agrupa os usuários em **personas** acionáveis para produto e negócio.

Tudo isso é entregue em um **dashboard web interativo** (Dash + Plotly), publicado em produção, e sustentado por notebooks de pesquisa e artefatos de modelo versionados no repositório.

> O repositório reúne o dashboard principal, as análises exploratórias em notebooks, os modelos treinados (classificação e clusterização) e os scripts de apoio à preparação de dados.

---

## 🖥️ O Dashboard

A aplicação principal vive em [`dash/app.py`](dash/app.py) e está publicada em:

### 👉 **[louquinhosbr-new-thinkmoney-dashboard.hf.space](https://louquinhosbr-new-thinkmoney-dashboard.hf.space/)**

É a forma mais rápida de explorar o projeto — sem instalar nada.

### Navegação

| Seção | O que oferece |
|---|---|
| 🏠 **Home** | Apresentação do projeto e identidade visual ThinkMoney. |
| 📚 **Documentação** | Descrição completa do dataset, com cards para variáveis categóricas e numéricas e resumo estatístico da base. |
| 📊 **EDA** | Análise exploratória com **filtros globais** por *City Tier*, *Occupation* e faixa de renda. |
| 🤖 **Machine Learning** | Predição de vulnerabilidade financeira com **threshold ajustável** e visualização em cards, gráfico de pizza e histograma. |

### Subseções da EDA no dashboard

| Código | Tema |
|---|---|
| `5.1` | **Waste Ratio** — desperdício financeiro |
| `5.2` | **Transporte** |
| `5.3` | **Condicionantes** da saúde financeira |
| `5.4` | **Moradia** |
| `5.5` | **Cansaço / Dopamina** — comportamento de consumo |


---

## 🗂️ O Dataset

Base financeira tabular, **sem valores nulos** (auditoria de qualidade confirmou 0% de ausências), com:

<div align="center">

| 📈 Registros | 🧮 Colunas | 🔢 Tipos | 🎯 Domínio |
|:---:|:---:|:---:|:---:|
| **20.000** | **27** | Numérico + Categórico | Finanças pessoais |

</div>

As variáveis estão organizadas em quatro grupos (detalhe completo em [`dataset/dicionario.txt`](dataset/dicionario.txt)):

**1. Renda e demografia** — `Income`, `Age`, `Dependents`, `Occupation`, `City_Tier`

**2. Despesas mensais** — `Rent`, `Loan_Repayment`, `Insurance`, `Groceries`, `Transport`, `Eating_Out`, `Entertainment`, `Utilities`, `Healthcare`, `Education`, `Miscellaneous`

**3. Metas e indicadores** — `Desired_Savings_Percentage`, `Desired_Savings`, `Disposable_Income`

**4. Economia potencial** — `Potential_Savings_*` (Groceries, Transport, Eating_Out, Entertainment, Utilities, Healthcare, Education, Miscellaneous)

### Features derivadas

O pré-processamento em [`dash/data.py`](dash/data.py) cria variáveis essenciais para as análises e modelos, normalizadas pela renda para eliminar efeito de escala:

`Total_Expenses` · `Savings_Rate` · `Waste_Ratio` · `Age_Group` · `Age_Group_5` · `Fixed_Costs` · `Fixed_Ratio` · `Disposable_Income` · `Rent_Income_Ratio`

---

## 🔍 Análises Exploratórias (EDA)

A pasta [`analises/`](analises/) concentra **11 investigações temáticas** em notebooks, cada uma atacando uma dimensão do problema financeiro:

| Análise | Foco |
|---|---|
| `analise-anomalias-financeiras` | Detecção de outliers e padrões atípicos de gasto |
| `analise-ciclo-de-vida-e-perfil-financeiro` | Como o perfil financeiro evolui com a idade |
| `analise-city-tier` | Impacto do porte da cidade nas finanças |
| `analise-custo-de-vida-e-city-tier` | Custo de vida cruzado com *City Tier* |
| `analise-gastos-e-poupanca` | Relação entre despesas e capacidade de poupar |
| `analise-independencia-de-variaveis-na-boa-gestao` | Quais variáveis explicam boa gestão financeira |
| `analise-insurance-age` | Comportamento de seguros por faixa etária |
| `analise-poupanca` | Viabilidade de metas e potencial de economia |
| `analise-saude-financeira-comprometimento-renda` | Comprometimento de renda e saúde financeira |
| `analise-simulacao-migracao` | Simulação de migração entre perfis |
| `analises-dopamina` | Consumo por impulso (cidade, idade e ocupação) |

### 💡 Principais insights

- **Base sólida:** 20.000 registros, **0% de valores nulos**.
- **Metas realistas:** apenas **0,56%** da base possui metas de poupança financeiramente inviáveis.
- **Renda × Renda disponível:** correlação positiva forte (**0,88**) — os gastos essenciais escalam proporcionalmente aos ganhos.
- **Maior oportunidade de economia:** a categoria **Supermercado (Groceries)** lidera o potencial médio de economia.
- **Consumo por idade:** gastos com *Eating Out* são uniformes entre faixas etárias, mas o grupo **18–25 anos** tem a maior média.
- **Outliers relevantes:** **1.257** em aluguel (concentrados em cidades Tier 2) e **1.326** em alimentação fora (liderados por *Professionals*).
- **Para modelagem:** `Rent` e `Groceries` são os melhores preditores de renda; `Loan_Repayment` e `Insurance` são os principais detratores da saúde financeira; há multicolinearidade entre as colunas `Potential_Savings_*` (justifica PCA / seleção de features).

---

## 🤖 Modelos de Machine Learning

O projeto entrega **dois modelos complementares**, ambos em [`modelos/`](modelos/):

### 🎯 1. Classificação — Identificação de Clientes Seguros (Baixo Risco de Crédito)

> **Objetivo:** encontrar os usuários com **baixo risco de crédito** — os clientes **`Seguro`** — separando-os dos que estão em situação de vulnerabilidade financeira. É por isso que toda a avaliação do modelo (Precisão, Recall, F1 e a análise SHAP) é orientada à classe `Seguro`: o interesse do negócio é identificar, com confiança, quem é um bom pagador / cliente saudável.

Modelo **supervisionado** que classifica cada usuário em **`Seguro`** (baixo risco) ou **`Vulnerável`** (alto risco). O rótulo `Vulnerable` é construído por regra de negócio: o usuário é considerado **vulnerável** quando atende a **pelo menos 2 de 4 critérios** de risco — caso contrário, é classificado como **seguro**:

- comprometimento elevado com empréstimos (`Loan_Repayment / Income > 10%`);
- alto gasto com itens não-essenciais (`(Eating_Out + Entertainment) / Income > 8,5%`);
- margem disponível insuficiente após a meta de poupança (`< 10%` da renda);
- alto potencial de economia em supermercado (`Potential_Savings_Groceries / Income > 8%`).

As *features* são percentuais normalizados pela renda (`Rent_Ratio`, `Healthcare_Ratio`, `Education_Ratio`, `Groceries_Ratio`, `Transport_Ratio`, `Utilities_Ratio`, `Insurance_Ratio`) somados a `Age` e `Dependents`, sem variáveis que vazem o alvo (*data leakage*).

**Benchmark de algoritmos** (notebook [`modelo_classificacao_regressao.ipynb`](modelos/mecanismo_vulnerabilidade/modelo_classificacao_regressao.ipynb)) — todos com pesos de classe ajustados para o desbalanceamento:

| Algoritmo | AUC-ROC |
|---|:---:|
| **Regressão Logística** | **0,653** |
| Gradient Boosting | 0,651 |
| SVM | 0,643 |
| CatBoost | 0,631 |
| Random Forest | 0,621 |
| XGBoost | 0,614 |

A **Regressão Logística** é o modelo de referência da solução, pela combinação de desempenho competitivo e alta interpretabilidade. A análise no notebook é complementada por:

- 📊 **SHAP** — importância das features para a classe `Seguro` e visão **global** (beeswarm), além de **curva de Pareto** das variáveis mais influentes;
- 🔁 **Cross-validation** estratificado (5 folds), com foco na classe `Seguro`: Recall médio **0,794** e Precisão média **0,827**;
- 🧩 **Matriz de confusão** e **tabela comparativa** dos seis modelos, ordenada por F1-Score;
- 🎯 **Segmentação de clientes** por probabilidade de ser `Seguro` (perfis de alta, moderada e baixa confiança).

### 🧩 2. Clusterização — Personas de Clientes (K-Means++)

Modelo **não supervisionado** que segmenta os usuários em **personas** de negócio. Todo o pipeline está no notebook [`clusterizacao_personas_final.Ipynb`](modelos/modelo_clusterizacao/clusterizacao_personas_final.Ipynb), organizado em blocos: preparação dos dados, exploração com **DBSCAN** e o modelo final **K-Means++** (com uma comparação direta entre os dois ao final).

- **Feature engineering:** 10 variáveis proporcionais à renda (*ratio features*) — `Rent_pct`, `Loan_Repayment_pct`, `Disposable_pct`, `Desired_Savings_pct`, `Gap_Poupanca`, `Gastos_Consumo_pct`, `Gastos_Fixos_pct`, `City_Tier_enc`, `Age` e `Dependents` — eliminando a dominância do `Income`;
- **Normalização:** `RobustScaler` (robusto a outliers, baseado em mediana e IQR);
- **Redução de dimensionalidade:** `PCA` configurado para reter **≥85% da variância** (`n_components=0.85`), resultando em ~92% na prática (6 componentes);
- **Seleção de *k*:** método do cotovelo e *Silhouette Score* para K de 2 a 10, com *Silhouette Diagram* detalhado para K ∈ {3, 4, 5};
- **Modelo final:** **K-Means++ com K=5**, escolhido pela melhor combinação de coesão e leitura de negócio. Métricas: **Silhouette ≈ 0,222** e **Davies-Bouldin ≈ 1,504**. O pipeline completo (scaler, PCA, modelo, personas e métricas) é salvo em [`modelo_kmeans_personas.pkl`](modelos/modelo_clusterizacao/modelo_kmeans_personas.pkl).

**As 5 personas identificadas** (nomeadas sob a ótica de risco de crédito):

| Persona | Volume | Perfil e risco de crédito |
|---|:---:|---|
| 🔴 **0 — O Estrangulado Financeiro** | 9,7% | Renda livre quase nula (7,8%) somada a dívidas altas, em centros Tier 1. **Risco altíssimo** — crédito novo tende à inadimplência. |
| 🟢 **1 — O Poupador Natural** | 40,3% | Maior renda disponível (33,9%), sem dívidas relevantes, poupa naturalmente. **Risco baixíssimo (prime)** — aprovação com limites altos. |
| 🟡 **2 — O Refém do Aluguel** | 19,0% | Aluguel ≈30% da renda (Tier 1) e quase nenhuma dívida. **Risco moderado** — bons pagadores, mas com despesa fixa pesada. |
| 🟠 **3 — O Endividado** | 20,3% | Empréstimos (13,8% da renda) corroem o orçamento, apesar do custo de vida menor. **Risco moderado a alto (alavancado).** |
| 🔵 **4 — O Estrategista** | 10,7% | Meta de poupança alta (17,4%) e disciplina para cumpri-la. **Risco baixíssimo (super prime)** — usa crédito de forma planejada. |

Análise completa das personas, com justificativa SHAP e recomendações de concessão de crédito, em [`analise_credito_personas.md`](modelos/modelo_clusterizacao/analise_credito_personas.md). A interpretabilidade no notebook usa um **surrogate Random Forest** com **SHAP** (`TreeExplainer`), curva de **Pareto** e **radar charts** por persona.

---

## 🧱 Stack utilizada

<div align="center">

`Python` · `Dash` · `Flask` · `Plotly` · `Pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `CatBoost` · `SHAP` · `Joblib` · `Gunicorn`

</div>

As dependências do dashboard estão em [`dash/requirements.txt`](dash/requirements.txt).

---

## ▶️ Como executar

### 1. Pré-requisitos

- Python **3.10+**
- `pip` atualizado

### 2. Clonar e criar o ambiente virtual

```bash
git clone <url-do-repositorio>
cd MachineLearning_PISI3
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependências

```bash
pip install -r dash/requirements.txt
```

### 4. Rodar o dashboard

> ⚠️ Execute **a partir da raiz do repositório** — a aplicação usa imports relativos ao diretório `dash`.

```bash
python dash/app.py
```

Acesse o endereço exibido no terminal, normalmente **http://127.0.0.1:8050/**.

---

## 📁 Estrutura do projeto

<details>
<summary><b>Ver a organização dos arquivos</b></summary>

```text
MachineLearning_PISI3/
├── README.md
├── LICENSE
├── Dockerfile
├── dataset/                          # Base de dados e dicionário
│   ├── data.csv
│   ├── data.parquet
│   └── dicionario.txt
├── dash/                             # 🖥️ Aplicação web (Dash)
│   ├── app.py                        #   ponto de entrada
│   ├── data.py                       #   carga e features derivadas
│   ├── requirements.txt
│   ├── assets/                       #   CSS e arquivos estáticos
│   ├── components/                   #   layout e home
│   ├── pages/                        #   home, documentação, análise, ml
│   └── sections/                     #   seções temáticas da EDA
├── analises/                         # 🔍 11 notebooks de EDA temática
│   ├── analise-anomalias-financeiras/
│   ├── analise-ciclo-de-vida-e-perfil-financeiro/
│   ├── analise-city-tier/
│   ├── analise-custo-de-vida-e-city-tier/
│   ├── analise-gastos-e-poupanca/
│   ├── analise-independencia-de-variaveis-na-boa-gestao/
│   ├── analise-insurance-age/
│   ├── analise-poupanca/
│   ├── analise-saude-financeira-comprometimento-renda/
│   ├── analise-simulacao-migracao/
│   └── analises-dopamina/
├── modelos/                          # 🤖 Modelagem
│   ├── mecanismo_vulnerabilidade/    #   classificação (benchmark + SHAP)
│   ├── modelo_clusterizacao/         #   K-Means++ de personas
│   ├── modelo_previsao_vulnerabilidade/
│   └── modelo_joao_vulnerabilidade/  #   modelo servido no dashboard
└── scripts/
    └── converter_para_parquet.py     # CSV → Parquet
```

</details>

---

## 🛠️ Script auxiliar

[`scripts/converter_para_parquet.py`](scripts/converter_para_parquet.py) converte `dataset/data.csv` em `dataset/data.parquet`, otimizando a leitura e a experimentação local.

---

## 📝 Observações de uso

- Rode sempre a partir da **raiz** do repositório (`python dash/app.py`).
- A seção de ML do dashboard depende dos artefatos `.pkl` (modelo + scaler) presentes na pasta do modelo servido.
- Ao substituir o dataset, revise a lógica de EDA e dos modelos para garantir que as colunas esperadas continuem existindo.

---

## 📜 Licença

Distribuído sob a licença descrita em [`LICENSE`](LICENSE).

<div align="center">

---

**Projeto acadêmico — PISI3 · Grupo 5**

*Feito com dados, Python e ☕*

</div>
