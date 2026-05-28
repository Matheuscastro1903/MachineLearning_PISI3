import pandas as pd
import os

def load_and_preprocess_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'data.csv')
    
    df = pd.read_csv(csv_path)
    
    df = df.dropna()
    
    # Criação de Colunas Derivadas
    expense_columns = [
        'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 
        'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 
        'Healthcare', 'Education', 'Miscellaneous'
    ]
    
    # Total Expenses = soma das colunas de despesas linha a linha
    df['Total_Expenses'] = df[expense_columns].sum(axis=1)

    # Savings Rate = o quanto a pessoa guarda da própria renda
    df['Savings_Rate'] = (df['Income'] - df['Total_Expenses']) / df['Income'].replace(0, 1)

    # Waste Ratio (seção 5.1): % da renda gasta em itens evitáveis
    waste_columns = ['Eating_Out', 'Entertainment', 'Miscellaneous']
    df['Waste_Ratio'] = (df[waste_columns].sum(axis=1) / df['Income'].replace(0, 1)) * 100

    # Faixa etária para análise da seção 5.1
    bins   = [17, 30, 45, 60, 200]
    labels = ['Jovem Adulto', 'Adulto', 'Sênior', 'Idoso']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    # Faixa etária 5 grupos para seção 5.3
    bins5   = [17, 25, 35, 45, 55, 200]
    labels5 = ['18-25', '26-35', '36-45', '46-55', '55+']
    df['Age_Group_5'] = pd.cut(df['Age'], bins=bins5, labels=labels5, right=True)

    # Gastos fixos e derivados para seção 5.3 / 5.6
    fixed_cols = ['Rent', 'Loan_Repayment', 'Insurance']
    df['Fixed_Costs']       = df[fixed_cols].sum(axis=1)
    df['Fixed_Ratio']       = df['Fixed_Costs'] / df['Income'].replace(0, 1)
    df['Disposable_Income'] = df['Income'] - df['Fixed_Costs']

    # Rent como proporção da renda para seção 5.4
    df['Rent_Income_Ratio'] = df['Rent'] / df['Income'].replace(0, 1)

    return df

# Exportamos um DataFrame global limpo para ser usado na inicialização do App
df_master = load_and_preprocess_data()