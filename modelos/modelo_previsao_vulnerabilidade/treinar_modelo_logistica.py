"""
Script de Treinamento — Modelo de Regressão Logística
Vulnerabilidade Financeira de Clientes

Pipeline:
  1. Carrega dataset (20.000 registros) em formato .parquet
  2. Filtra Desired_Savings > 0  →  19.888 registros
  3. Engenharia de features e construção do risk_score
  4. Define target: Vulnerable = (risk_score >= 2)
  5. Treina LogisticRegression com as 11 features finais
  6. Salva modelo e scaler em .pkl
"""

import os
import pathlib
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ─────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
ML_DIR     = BASE_DIR.parents[1]           # MachineLearning_PISI3/
DATA_PATH  = ML_DIR / "dataset" / "data.parquet"
MODEL_OUT  = BASE_DIR / "modelo_regressao_logistica.pkl"
SCALER_OUT = BASE_DIR / "scaler_logistica.pkl"

# ─────────────────────────────────────────────
# 1. Carregamento
# ─────────────────────────────────────────────
print("Carregando dataset...")
df = pd.read_parquet(DATA_PATH)
print(f"  Registros originais : {len(df):,}  |  Colunas: {df.shape[1]}")

# ─────────────────────────────────────────────
# 2. Filtro: Desired_Savings > 0  →  19.888
# ─────────────────────────────────────────────
n_zero = (df['Desired_Savings'] == 0).sum()
print(f"  Registros com Desired_Savings == 0 removidos: {n_zero}")
df = df[df['Desired_Savings'] > 0].reset_index(drop=True)
print(f"  Registros após filtro: {len(df):,}")

# ─────────────────────────────────────────────
# 3. Engenharia de Features (para o risk_score)
# ─────────────────────────────────────────────
# % da renda gasto em entretenimento + refeições fora
df['perc_nao_essenciais'] = (df['Eating_Out'] + df['Entertainment']) / df['Income']

# % da renda comprometida com empréstimo
df['perc_emprestimo'] = df['Loan_Repayment'] / df['Income']

# Total de poupança potencial não realizada
colunas_potencial = [
    'Potential_Savings_Groceries',
    'Potential_Savings_Transport',
    'Potential_Savings_Eating_Out',
    'Potential_Savings_Entertainment',
    'Potential_Savings_Utilities',
    'Potential_Savings_Healthcare',
    'Potential_Savings_Education',
    'Potential_Savings_Miscellaneous',
]
df['total_potential_savings']  = df[colunas_potencial].sum(axis=1)
df['perc_potential_savings']   = df['total_potential_savings'] / df['Income']

# Margem de segurança após poupança desejada
df['buffer_emergencia'] = (df['Disposable_Income'] - df['Desired_Savings']) / df['Income']

# ─────────────────────────────────────────────
# 4. Risk Score e Target
# ─────────────────────────────────────────────
c1 = (df['perc_emprestimo']      > 0.10).astype(int)   # empréstimo pesado
c2 = (df['perc_nao_essenciais']  > 0.085).astype(int)  # gastos não essenciais altos
c3 = (df['buffer_emergencia']    < 0.10).astype(int)   # buffer de emergência baixo
c4 = (df['perc_potential_savings'] > 0.08).astype(int) # poupança potencial desperdiçada

df['risk_score'] = c1 + c2 + c3 + c4
df['Vulnerable'] = (df['risk_score'] >= 2).astype(int)

print(f"\nDistribuição do target:")
print(df['Vulnerable'].value_counts().to_string())

# ─────────────────────────────────────────────
# 5. Features Finais (11 variáveis)
# ─────────────────────────────────────────────
FEATURES = [
    'Income',
    'Age',
    'Dependents',
    'Loan_Repayment',
    'Eating_Out',
    'Entertainment',
    'Healthcare',
    'Rent',
    'Groceries',
    'Disposable_Income',
    'Desired_Savings',
]

X = df[FEATURES]
y = df['Vulnerable']

print(f"\nFeatures ({len(FEATURES)}): {FEATURES}")

# ─────────────────────────────────────────────
# 6. Split Treino / Teste
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTreino : {len(X_train):,} registros")
print(f"Teste  : {len(X_test):,} registros")

# ─────────────────────────────────────────────
# 7. Normalização
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 8. Treinamento — Regressão Logística
# ─────────────────────────────────────────────
print("\nTreinando Regressão Logística...")
lr = LogisticRegression(
    random_state=42,
    class_weight={0: 1, 1: 4},
    max_iter=1000,
)
lr.fit(X_train_scaled, y_train)
print("  Treinamento concluído!")

# ─────────────────────────────────────────────
# 9. Avaliação
# ─────────────────────────────────────────────
y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]
auc    = roc_auc_score(y_test, y_prob)

print(f"\nAUC-ROC: {auc:.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Seguro', 'Vulnerável']))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Matriz de Confusão:")
print(f"  Verdadeiros Negativos  (Seguro → Seguro)       : {tn}")
print(f"  Falsos Positivos       (Seguro → Vulnerável)   : {fp}")
print(f"  Falsos Negativos       (Vulnerável → Seguro)   : {fn}")
print(f"  Verdadeiros Positivos  (Vulnerável → Vulnerável): {tp}")
print(f"\n  De {tp+fn} vulneráveis reais → {tp} identificados ({tp/(tp+fn)*100:.1f}%)")

# ─────────────────────────────────────────────
# 10. Segmentação por Nível de Risco (base completa)
# ─────────────────────────────────────────────
X_all_scaled           = scaler.transform(X)
df['prob_risco']       = lr.predict_proba(X_all_scaled)[:, 1]

def classificar_risco(prob):
    if prob < 0.30:
        return 'Baixo'
    elif prob < 0.60:
        return 'Médio'
    else:
        return 'Alto'

df['nivel_risco'] = df['prob_risco'].apply(classificar_risco)

seg = df.groupby('nivel_risco').agg(
    clientes=('Income', 'count'),
    renda_media=('Income', 'mean'),
    emprestimo_medio=('Loan_Repayment', 'mean'),
    prob_media=('prob_risco', 'mean'),
).reindex(['Baixo', 'Médio', 'Alto'])
seg['pct_base']         = (seg['clientes'] / len(df) * 100).round(1)
seg['exposicao_mensal'] = seg['clientes'] * seg['emprestimo_medio']

print("\n" + "=" * 60)
print("  SEGMENTAÇÃO POR NÍVEL DE RISCO")
print("=" * 60)
for nivel in ['Baixo', 'Médio', 'Alto']:
    r = seg.loc[nivel]
    print(f"\n  Risco {nivel}:")
    print(f"    Clientes          : {int(r['clientes']):,}")
    print(f"    % da base         : {r['pct_base']}%")
    print(f"    Prob. média       : {r['prob_media']*100:.1f}%")
    print(f"    Empréstimo médio  : R$ {r['emprestimo_medio']:,.0f}")
    print(f"    Exposição mensal  : R$ {r['exposicao_mensal']:,.0f}")

# ─────────────────────────────────────────────
# 11. Persistência — .pkl
# ─────────────────────────────────────────────
payload = {
    "model":    lr,
    "scaler":   scaler,
    "features": FEATURES,
    "auc_roc":  round(auc, 4),
    "target":   "Vulnerable",
    "threshold_risk": 2,        # risk_score >= 2 → Vulnerável
    "class_weight": {0: 1, 1: 4},
}

joblib.dump(payload, MODEL_OUT)
print(f"\n✅ Modelo salvo  : {MODEL_OUT}")

# Scaler também disponível isoladamente (para uso no dash/pipeline)
joblib.dump(scaler, SCALER_OUT)
print(f"✅ Scaler salvo  : {SCALER_OUT}")
print("\nPronto!")
