# MachineLearning_PISI3

Projeto de análise de dados e Machine Learning com foco em finanças pessoais, construído com Dash, Pandas, Plotly e Scikit-Learn. O repositório reúne o dashboard principal ThinkMoney, análises exploratórias em notebooks, modelos treinados e scripts de apoio para preparação de dados.

## Visão geral

O objetivo do projeto é transformar uma base de dados financeira em uma experiência interativa de exploração e previsão. A aplicação permite navegar entre:

- uma página inicial de apresentação do projeto;
- uma área de documentação técnica do dataset;
- uma seção de EDA com filtros globais e análises temáticas;
- uma seção de Machine Learning para estimar vulnerabilidade financeira.

Além do dashboard, o repositório também contém notebooks de estudo e artefatos de modelagem usados no processo analítico.

## Dashboard publicado

O dashboard está disponível em produção no link abaixo:

- https://dashboard-pisi3-grupo-5.onrender.com/

Essa é a forma mais rápida de acessar a versão já publicada da aplicação.

## Principais funcionalidades

- Dashboard web em Dash com navegação interna por botões.
- Página de documentação com resumo estatístico e descrição das variáveis.
- EDA com filtros por cidade, ocupação e faixa de renda.
- Seções analíticas sobre desperdício financeiro, transporte, moradia, comportamento e vulnerabilidade.
- Modelo de Machine Learning com ajuste de sensibilidade por threshold.
- Script utilitário para converter o dataset de CSV para Parquet.

## Estrutura do projeto

```text
MachineLearning_PISI3/
├── README.md
├── LICENSE
├── dataset/
│   ├── data.csv
│   └── dicionario.txt
├── dash/
│   ├── app.py
│   ├── data.py
│   ├── requirements.txt
│   ├── assets/
│   │   └── style.css
│   ├── components/
│   ├── pages/
│   ├── sections/
│   └── dataset/
├── analises/
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
├── modelos/
│   ├── modelo_clusterizacao/
│   └── modelo_joao_vulnerabilidade/
└── scripts/
	└── converter_para_parquet.py
```

## Stack utilizada

- Python
- Dash
- Flask
- Pandas
- NumPy
- Plotly
- Scikit-Learn
- Joblib

As dependências do dashboard estão listadas em `dash/requirements.txt`.

## Dataset

O projeto trabalha com uma base financeira tabular que, pela documentação interna do dashboard, possui:

- 20.000 registros;
- 27 colunas;
- variáveis numéricas e categóricas;
- campos relacionados a renda, despesas mensais, economia potencial e características demográficas.

Entre as variáveis descritas na aplicação estão:

- Income
- Age
- Dependents
- Rent
- Loan_Repayment
- Groceries
- Transport
- Eating_Out
- Entertainment
- Insurance
- Utilities
- Healthcare
- Education
- Miscellaneous
- Occupation
- City_Tier

O processamento em `dash/data.py` cria colunas derivadas importantes para as análises, como Total_Expenses, Savings_Rate, Waste_Ratio, Age_Group, Age_Group_5, Fixed_Costs, Fixed_Ratio, Disposable_Income e Rent_Income_Ratio.

## Como executar

### 1. Pré-requisitos

- Python 3.10 ou superior
- pip atualizado

### 2. Criar e ativar um ambiente virtual

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```powershell
pip install -r dash\requirements.txt
```

### 4. Executar a aplicação

A aplicação principal fica em `dash/app.py`. Execute a partir da raiz do projeto com:

```powershell
python dash\app.py
```

Depois disso, abra o endereço exibido no terminal, normalmente `http://127.0.0.1:8050/`.

## Funcionalidades do dashboard

### Home

A tela inicial apresenta a identidade visual do projeto ThinkMoney e direciona o usuário para a análise exploratória ou para o modelo preditivo.

### Documentação

A página de documentação detalha o dataset com cards explicativos para variáveis categóricas e numéricas, além de um resumo geral da base.

### EDA

A seção de análise exploratória permite filtrar a base por:

- City Tier;
- Occupation;
- faixa de renda.

As subseções disponíveis são:

- 5.1 - Waste Ratio
- 5.2 - Transporte
- 5.3 - Condicionantes
- 5.4 - Moradia
- 5.5 - Cansaço/Dopamina
- 5.6 - M1 (Vulnerabilidade)

### Machine Learning

A seção 5.6 executa uma predição de vulnerabilidade financeira com um modelo treinado. O usuário pode ajustar o threshold para tornar a classificação mais conservadora ou mais permissiva, e visualizar os resultados em cards, gráfico de pizza e histograma.

## Artefatos de modelo

O módulo de vulnerabilidade financeira espera os seguintes arquivos dentro de `modelos/modelo_joao_vulnerabilidade/`:

- `modelo_vulnerabilidade.pkl`
- `scaler.pkl`

Sem esses dois artefatos, a seção de ML não consegue executar a predição.

## Script auxiliar

O arquivo `scripts/converter_para_parquet.py` converte `dataset/data.csv` para `dataset/data.parquet`, o que pode ser útil para otimização de leitura e experimentação local.

## Organização das análises

A pasta `analises/` concentra notebooks de exploração e investigação temática do problema. Já a pasta `modelos/` reúne os materiais de modelagem, incluindo experimentos de clusterização e a solução usada para a previsão de vulnerabilidade.

## Observações de uso

- O projeto foi organizado para ser executado a partir da raiz do repositório usando `python dash\app.py`.
- A aplicação utiliza imports relativos ao diretório `dash`, então executar o arquivo de entrada a partir de outro local pode exigir ajustes no caminho.
- Se o dataset for substituído, a lógica de EDA e do modelo deve ser revisada para garantir que as colunas esperadas continuem existindo.


