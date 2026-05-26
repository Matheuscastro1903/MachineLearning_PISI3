# ==============================================================================
# MODELO DE PREVISÃO DE VULNERABILIDADE FINANCEIRA
# Como usar: python previsao_vulnerabilidade.py
#
# REQUISITOS:
#   - modelo_vulnerabilidade.pkl  (gerado no Jupyter)
#   - scaler.pkl                  (gerado no Jupyter)
#   - arquivo CSV ou parquet com os dados dos clientes
#
# COLUNAS NECESSÁRIAS NO ARQUIVO DE DADOS:
#   Income, Age, Dependents, Loan_Repayment, Eating_Out,
#   Entertainment, Healthcare, Rent, Groceries,
#   Disposable_Income, Desired_Savings
# ==============================================================================

import pandas as pd
import joblib
import os
import sys

# ==============================================================================
# CONFIGURAÇÃO — ALTERE AQUI
# ==============================================================================

CAMINHO_MODELO  = 'modelo_vulnerabilidade.pkl'   # ← modelo treinado
CAMINHO_SCALER  = 'scaler.pkl'                   # ← normalizador
CAMINHO_DADOS   = 'data.parquet'                 # ← seu arquivo de dados (.csv ou .parquet)
CAMINHO_SAIDA   = 'resultado_risco.csv'          # ← onde salvar o resultado

# ==============================================================================

FEATURES = [
    'Income', 'Age', 'Dependents', 'Loan_Repayment',
    'Eating_Out', 'Entertainment', 'Healthcare',
    'Rent', 'Groceries', 'Disposable_Income', 'Desired_Savings'
]

# ------------------------------------------------------------------------------
def separador(titulo):
    print("\n" + "=" * 60)
    print(f"  {titulo}")
    print("=" * 60)

# ------------------------------------------------------------------------------
def classificar_risco(prob):
    """Converte probabilidade em nível de risco."""
    if prob < 0.30:
        return 'Baixo'
    elif prob < 0.60:
        return 'Médio'
    else:
        return 'Alto'

# ------------------------------------------------------------------------------
def carregar_modelo():
    """Carrega o modelo e o scaler salvos."""
    for arquivo in [CAMINHO_MODELO, CAMINHO_SCALER]:
        if not os.path.exists(arquivo):
            print(f"\n❌ ERRO: Arquivo '{arquivo}' não encontrado.")
            print("   Certifique-se de que rodou o bloco de salvamento no Jupyter.")
            sys.exit(1)

    modelo = joblib.load(CAMINHO_MODELO)
    scaler = joblib.load(CAMINHO_SCALER)
    print(f"\n✅ Modelo carregado: {CAMINHO_MODELO}")
    print(f"✅ Scaler carregado: {CAMINHO_SCALER}")
    return modelo, scaler

# ------------------------------------------------------------------------------
def carregar_dados():
    """Carrega o arquivo de dados (CSV ou parquet)."""
    if not os.path.exists(CAMINHO_DADOS):
        print(f"\n❌ ERRO: Arquivo de dados '{CAMINHO_DADOS}' não encontrado.")
        sys.exit(1)

    if CAMINHO_DADOS.endswith('.parquet'):
        df = pd.read_parquet(CAMINHO_DADOS)
    elif CAMINHO_DADOS.endswith('.xlsx') or CAMINHO_DADOS.endswith('.xls'):
        df = pd.read_excel(CAMINHO_DADOS)
    else:
        try:
            df = pd.read_csv(CAMINHO_DADOS, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(CAMINHO_DADOS, encoding='latin-1')

    print(f"✅ Dados carregados: {len(df):,} clientes | {df.shape[1]} colunas")
    return df

# ------------------------------------------------------------------------------
def validar_colunas(df):
    """Verifica se todas as colunas necessárias existem."""
    faltando = [col for col in FEATURES if col not in df.columns]
    if faltando:
        print(f"\n❌ ERRO: Colunas ausentes no arquivo de dados:")
        for col in faltando:
            print(f"   - {col}")
        sys.exit(1)
    print(f"✅ Todas as {len(FEATURES)} colunas necessárias encontradas.")

# ------------------------------------------------------------------------------
def prever(df, modelo, scaler):
    """Aplica o modelo e retorna DataFrame com previsões."""
    X = df[FEATURES].copy()
    X_scaled = scaler.transform(X)

    df = df.copy()
    df['probabilidade_risco'] = modelo.predict_proba(X_scaled)[:, 1]
    df['classificacao_binaria'] = modelo.predict(X_scaled)   # 0 = Seguro, 1 = Vulnerável
    df['nivel_risco'] = df['probabilidade_risco'].apply(classificar_risco)

    return df

# ------------------------------------------------------------------------------
def exibir_resumo(df):
    """Exibe o resumo da segmentação por nível de risco."""
    separador("RESULTADO DA ANÁLISE")

    total = len(df)
    for nivel in ['Baixo', 'Médio', 'Alto']:
        grupo = df[df['nivel_risco'] == nivel]
        n = len(grupo)
        pct = n / total * 100
        prob_media = grupo['probabilidade_risco'].mean() * 100
        emp_medio = grupo['Loan_Repayment'].mean()
        exposicao = grupo['Loan_Repayment'].sum()

        emoji = {'Baixo': '🟢', 'Médio': '🟡', 'Alto': '🔴'}[nivel]
        print(f"\n{emoji} Risco {nivel}:")
        print(f"   Clientes:           {n:,} ({pct:.1f}% da base)")
        print(f"   Prob. média:        {prob_media:.1f}%")
        print(f"   Empréstimo médio:   R$ {emp_medio:,.0f}")
        print(f"   Exposição total:    R$ {exposicao:,.0f}")

    # Impacto financeiro
    separador("IMPACTO FINANCEIRO — GRUPO DE ALTO RISCO")
    alto = df[df['nivel_risco'] == 'Alto']
    exposicao_total = alto['Loan_Repayment'].sum()
    economia = exposicao_total * 0.30

    print(f"\n  Clientes em alto risco:    {len(alto):,}")
    print(f"  Exposição mensal total:    R$ {exposicao_total:,.0f}")
    print(f"  Economia potencial (30%):  R$ {economia:,.0f}")
    print(f"\n  * Estimativa conservadora de prevenção com ação antecipada")

    # Classificação binária
    separador("CLASSIFICAÇÃO BINÁRIA DO MODELO")
    vulneraveis = (df['classificacao_binaria'] == 1).sum()
    seguros = (df['classificacao_binaria'] == 0).sum()
    print(f"\n  Vulneráveis (1): {vulneraveis:,} clientes ({vulneraveis/len(df)*100:.1f}%)")
    print(f"  Seguros     (0): {seguros:,} clientes ({seguros/len(df)*100:.1f}%)")

# ------------------------------------------------------------------------------
def exportar(df):
    """Salva o resultado com as previsões em CSV."""
    colunas_saida = FEATURES + ['probabilidade_risco', 'classificacao_binaria', 'nivel_risco']
    df[colunas_saida].to_csv(CAMINHO_SAIDA, index=False)
    print(f"\n✅ Resultado exportado: {CAMINHO_SAIDA}")
    print(f"   {len(df):,} clientes analisados e salvos.")

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

if __name__ == '__main__':
    separador("MODELO DE VULNERABILIDADE FINANCEIRA")

    separador("CARREGANDO MODELO")
    modelo, scaler = carregar_modelo()

    separador("CARREGANDO DADOS")
    df = carregar_dados()

    separador("VALIDANDO COLUNAS")
    validar_colunas(df)

    separador("APLICANDO MODELO")
    df = prever(df, modelo, scaler)
    print(f"\n✅ Previsões geradas para {len(df):,} clientes.")

    exibir_resumo(df)
    exportar(df)

    separador("CONCLUÍDO")
    print("\n  Abra o arquivo 'resultado_risco.csv' para ver")
    print("  todos os clientes com seu nível de risco individual.\n")