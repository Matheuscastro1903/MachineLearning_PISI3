import pandas as pd
import os

def load_and_preprocess_data():
    # Ajuste o caminho do CSV de acordo com a sua pasta
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'data.csv')
    
    # 1. Carregamento
    df = pd.read_csv(csv_path)
    
    # 2. Limpeza básica (remover valores nulos se houver)
    df = df.dropna()
    
    # 3. Criação de Colunas Derivadas
    # Lista de colunas de despesas mapeadas do dicionário
    expense_columns = [
        'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 
        'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 
        'Healthcare', 'Education', 'Miscellaneous'
    ]
    
    # Total Expenses = soma das colunas de despesas linha a linha
    df['Total_Expenses'] = df[expense_columns].sum(axis=1)
    
    # Savings Rate = o quanto a pessoa guarda da própria renda
    # Vamos usar (Income - Total_Expenses) / Income. Ou, usando a coluna Disposable_Income se preferir.
    # Adicionando um pequeno valor na divisão para evitar divisão por zero, caso Income seja 0.
    df['Savings_Rate'] = (df['Income'] - df['Total_Expenses']) / df['Income'].replace(0, 1)
    
    return df

# Exportamos um DataFrame global limpo para ser usado na inicialização do App
df_master = load_and_preprocess_data()