# 📊 Análise de Personas para Modelagem de Crédito

Com base na aplicação do algoritmo **K-Means++**, explicabilidade via **SHAP (Princípio de Pareto)** e uma ótica voltada para **Risco e Concessão de Crédito**, os cinco clusters encontrados representam comportamentos financeiros muito distintos. 

Abaixo apresento a nomeação oficial de cada persona, a justificativa estatística (Pareto) e como um modelo de análise de crédito pode utilizar esses perfis para tomada de decisão (aprovação, limites e taxas).

---

## 🔴 Cluster 0: O Estrangulado Financeiro
**Tamanho:** ~9,7% da base

* **A Justificativa Pareto:** O modelo SHAP apontou que este grupo é definido pela sua drástica falta de **Renda Livre** (`Disposable_pct` com 22.3% de impacto) combinada com um alto peso de **Dívidas Atuais** (`Loan_Repayment_pct` com 21.0%). Eles são o único grupo cujo gap de poupança é positivo (não conseguem atingir a meta porque a renda acaba antes).
* **Visão de Análise de Crédito:**
  * **Perfil de Risco:** **Altíssimo (High Risk / Reject).**
  * **Comportamento:** A renda já está inteiramente comprometida pelo aluguel caro (Tier 1) e parcelas de empréstimos anteriores. A liquidez é quase nula (apenas 7,8% de sobra).
  * **Ação do Modelo:** Qualquer novo crédito concedido tem altíssima probabilidade de inadimplência (default). O modelo deve **negar novas linhas de crédito sem garantia** ou exigir colaterais rigorosos. A única oferta viável seria uma renegociação de dívida para baixar o peso das parcelas atuais.

---

## 🟢 Cluster 1: O Poupador Natural
**Tamanho:** ~40,3% da base (A grande massa)

* **A Justificativa Pareto:** Definidos quase inteiramente pelo **Gap de Poupança altamente negativo** (`Gap_Poupanca` com 34.3% de impacto). Isso significa que, independentemente da meta modesta que possuem, a renda livre deles (33.9%) engole as despesas. Eles poupam naturalmente pela ausência de gastos fixos altos.
* **Visão de Análise de Crédito:**
  * **Perfil de Risco:** **Baixíssimo (Prime / Low Risk).**
  * **Comportamento:** Vivem abaixo de suas possibilidades. Não possuem dívidas (`Loan` de 0.9%) e moram em cidades onde o custo de vida não os sufoca. 
  * **Ação do Modelo:** Excelentes candidatos para **aprovação automática com limites altos**. Como têm grande sobra de caixa mensal, o risco de calote é mínimo. O modelo pode direcioná-los para cartões de crédito premium (para centralizar os gastos que eles já têm) ou produtos de investimento (já que o dinheiro está sobrando).

---

## 🟡 Cluster 2: O Refém do Aluguel
**Tamanho:** ~19% da base

* **A Justificativa Pareto:** Pareto foi cirúrgico aqui, isolando a dupla **Aluguel** (`Rent_pct` com 30.9%) e **Tamanho da Cidade** (`City_Tier_enc` com 30.8%). O modelo ignorou a dívida, porque a identidade deles é o custo de moradia nas grandes metrópoles (Tier 1).
* **Visão de Análise de Crédito:**
  * **Perfil de Risco:** **Moderado com Vulnerabilidade Específica.**
  * **Comportamento:** Eles são bons pagadores hoje (têm quase zero dívida em empréstimos), mas têm uma despesa fixa altíssima e inegociável todo mês: o aluguel (30% da renda).
  * **Ação do Modelo:** O limite de crédito deve ser calculado com **cautela**, subtraindo o peso do aluguel da renda bruta. Eles são clientes ideais para **financiamento imobiliário** (já provaram que conseguem arcar com uma "parcela" pesada de moradia todo mês) ou para linhas de crédito que ofereçam fôlego no fluxo de caixa (ex: parcelamento de PIX/boletos de aluguel).

---

## 🟠 Cluster 3: O Endividado
**Tamanho:** ~20,3% da base

* **A Justificativa Pareto:** Diferente do C2 (que sofre com aluguel), o Pareto aponta que o grande vilão do Cluster 3 é a **Dívida** (`Loan_Repayment_pct` isolado no topo com 23.4% de impacto). O custo de vida deles é menor, mas as parcelas de crédito corroem o orçamento.
* **Visão de Análise de Crédito:**
  * **Perfil de Risco:** **Moderado a Alto (Alavancado / Overleveraged).**
  * **Comportamento:** Eles têm apetite por crédito e já o utilizaram (comprometendo 13.8% da renda com dívidas). Eles ainda têm alguma renda livre (19%), mas estão no limite saudável de alavancagem.
  * **Ação do Modelo:** Conceder *mais* crédito rotativo ou cartões pode empurrá-los para o status de "Estrangulado" (Cluster 0). O modelo de crédito deve identificá-los como alvo para **Consolidação de Dívidas (Portabilidade)**. A oferta ideal é um crédito com taxa menor que quite as dívidas antigas, alongando o prazo e melhorando a renda livre.

---

## 🔵 Cluster 4: O Estrategista
**Tamanho:** ~10,7% da base

* **A Justificativa Pareto:** As features que definem este grupo são puramente comportamentais: a **Meta de Poupança** (`Desired_Savings_pct` com 33%) e a capacidade de superá-la (`Gap_Poupanca` com 25.3%). Eles não são definidos pelo que gastam, mas pelo objetivo que traçam (meta altíssima de 17.4% da renda).
* **Visão de Análise de Crédito:**
  * **Perfil de Risco:** **Baixíssimo (Super Prime / Planejador).**
  * **Comportamento:** São disciplinados. Eles controlam o aluguel, evitam dívidas ruins e focam em acumular capital. Eles entendem de finanças pessoais.
  * **Ação do Modelo:** É o cliente mais seguro e rentável se o produto for bem posicionado. Eles não pegam crédito por desespero, mas podem usar crédito de forma *estratégica* (ex: financiar um carro para não descapitalizar os investimentos). O modelo deve oferecer a eles as **melhores taxas de juros (Pricing Risk-Based)** para atraí-los, focando em consórcios, financiamentos de bens duráveis ou crédito com garantia de investimentos.

---

### 📌 Resumo da Estratégia de Decisão de Crédito baseada no Modelo

| Cluster | Persona | Ação Principal de Crédito | Produto Alvo |
| :--- | :--- | :--- | :--- |
| **0** | **O Estrangulado** | **Negar / Restringir** | Renegociação de dívidas. |
| **1** | **O Poupador Natural** | **Aprovar (Auto)** | Cartão Black/Platinum, Investimentos. |
| **2** | **O Refém do Aluguel** | **Limitar Cautelosamente** | Financiamento Imobiliário. |
| **3** | **O Endividado** | **Monitorar** | Portabilidade de Crédito / Refinanciamento. |
| **4** | **O Estrategista** | **Aprovar (Auto) + Taxa Baixa**| Consórcio, Crédito com Garantia. |
