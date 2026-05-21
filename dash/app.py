import dash
from dash import dcc, html, Input, Output
import pandas as pd
import json

# Importamos nosso dataframe tratado do módulo data.py
from data import df_master

# Inicializando a aplicação Dash
app = dash.Dash(__name__)

# Pegando os valores únicos para popular os Dropdowns
city_tiers = df_master['City_Tier'].unique().tolist()
occupations = df_master['Occupation'].unique().tolist()
min_income = df_master['Income'].min()
max_income = df_master['Income'].max()

app.layout = html.Div([
    html.H1("Dashboard Financeiro - Exploratória"),
    
    # Componente dcc.Store guarda os dados filtrados na memória do navegador do usuário
    dcc.Store(id='filtered-data-store'),

    # Seção de Filtros (Nossa "Barra Lateral" ou Cabeçalho)
    html.Div([
        html.Label("Selecione o City Tier:"),
        dcc.Dropdown(
            id='filter-city',
            options=[{'label': city, 'value': city} for city in city_tiers],
            multi=True, # Permite selecionar mais de um
            placeholder="Todos os Tiers"
        ),

        html.Label("Selecione a Ocupação:"),
        dcc.Dropdown(
            id='filter-occupation',
            options=[{'label': occ, 'value': occ} for occ in occupations],
            multi=True,
            placeholder="Todas as Ocupações"
        ),

        html.Label("Faixa de Renda (Income):"),
        dcc.RangeSlider(
            id='filter-income',
            min=min_income,
            max=max_income,
            step=1000,
            marks={int(min_income): str(int(min_income)), int(max_income): str(int(max_income))},
            value=[min_income, max_income]
        )
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'marginBottom': '20px'}),

    # Placeholder para onde os gráficos entrarão depois
    html.Div(id='debug-output', style={'padding': '20px', 'border': '1px solid #ccc'})
])

# --- CALLBACK CENTRAL ---
# O Callback "escuta" as mudanças nos filtros e atualiza o dcc.Store
@app.callback(
    Output('filtered-data-store', 'data'),
    [Input('filter-city', 'value'),
     Input('filter-occupation', 'value'),
     Input('filter-income', 'value')]
)
def update_store(selected_cities, selected_occupations, income_range):
    # Começamos com uma cópia do dataframe completo
    dff = df_master.copy()
    
    # Filtro de Cidade
    if selected_cities:
        dff = dff[dff['City_Tier'].isin(selected_cities)]
        
    # Filtro de Ocupação
    if selected_occupations:
        dff = dff[dff['Occupation'].isin(selected_occupations)]
        
    # Filtro de Renda
    if income_range:
        dff = dff[(dff['Income'] >= income_range[0]) & (dff['Income'] <= income_range[1])]
    
    # O dcc.Store trafega dados no formato JSON. Convertemos o Pandas para um dicionário.
    return dff.to_dict('records')

# Callback de Teste (Apenas para vermos se o Store está recebendo os dados corretamente)
@app.callback(
    Output('debug-output', 'children'),
    Input('filtered-data-store', 'data')
)
def display_debug_info(stored_data):
    if not stored_data:
        return "Nenhum dado disponível."
    
    # Retorna quantas linhas sobraram após o filtro para testarmos na tela
    return f"Dados filtrados com sucesso! O dataset atual possui {len(stored_data)} registros."

if __name__ == '__main__':
    app.run(debug=True)