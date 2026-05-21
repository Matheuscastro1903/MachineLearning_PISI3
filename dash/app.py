import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from scipy.stats import kruskal

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
    html.Div(id='debug-output', style={'padding': '20px', 'border': '1px solid #ccc'}),

    html.Hr(),

    # ── SEÇÃO 5.1 ──────────────────────────────────────────────────────────────
    html.H2("5.1 – Waste Ratio por Variável Demográfica"),

    html.Div([
        html.Label("Agrupar por:"),
        dcc.RadioItems(
            id='s51-group-var',
            options=[
                {'label': ' Faixa Etária',         'value': 'Age_Group'},
                {'label': ' Ocupação Profissional', 'value': 'Occupation'},
                {'label': ' Nº de Dependentes',     'value': 'Dependents'},
            ],
            value='Age_Group',
            inline=True,
            inputStyle={'marginRight': '4px'},
            labelStyle={'marginRight': '20px'},
        ),
    ], style={'marginBottom': '15px'}),

    dcc.Graph(id='s51-violin'),

    html.Div(id='s51-kruskal', style={
        'padding': '10px',
        'border': '1px solid #ddd',
        'borderRadius': '4px',
        'marginTop': '8px',
        'fontFamily': 'monospace',
    }),
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

AGE_ORDER = ['Jovem Adulto', 'Adulto', 'Sênior', 'Idoso']

LABEL_MAP = {
    'Age_Group':  'Faixa Etária',
    'Occupation': 'Ocupação Profissional',
    'Dependents': 'Nº de Dependentes',
}


# ── CALLBACK SEÇÃO 5.1 ────────────────────────────────────────────────────────
@app.callback(
    Output('s51-violin',  'figure'),
    Output('s51-kruskal', 'children'),
    Input('filtered-data-store', 'data'),
    Input('s51-group-var', 'value'),
)
def update_s51(stored_data, group_var):
    if not stored_data:
        return {}, "Nenhum dado disponível."

    dff = pd.DataFrame(stored_data).dropna(subset=[group_var, 'Waste_Ratio']).copy()

    # Ordem categórica e conversão por variável
    if group_var == 'Age_Group':
        dff[group_var] = pd.Categorical(dff[group_var], categories=AGE_ORDER, ordered=True)
        cat_order = {group_var: AGE_ORDER}
    elif group_var == 'Dependents':
        dep_min = int(dff[group_var].min())
        dep_max = int(dff[group_var].max())
        dff[group_var] = dff[group_var].astype(int).astype(str)
        dep_order = [str(i) for i in range(dep_min, dep_max + 1)]
        cat_order = {group_var: dep_order}
    else:
        cat_order = {}

    label_x = LABEL_MAP[group_var]

    fig = px.violin(
        dff,
        x=group_var,
        y='Waste_Ratio',
        color=group_var,
        box=True,
        points='outliers',
        labels={'Waste_Ratio': '% de Economia sobre a Renda', group_var: label_x},
        category_orders=cat_order,
    )
    fig.update_layout(
        showlegend=False,
        height=420,
        margin=dict(t=30, b=50, l=60, r=20),
        yaxis_title='% de Economia sobre a Renda',
        xaxis_title=label_x,
    )

    # Kruskal-Wallis
    groups = [
        g['Waste_Ratio'].values
        for _, g in dff.groupby(group_var, observed=True)
        if len(g) >= 2
    ]
    if len(groups) >= 2:
        stat, pval = kruskal(*groups)
        sig = pval < 0.05
        conclusao = (
            "Diferença estatisticamente significativa (p < 0,05)"
            if sig else
            "Sem diferença significativa (p ≥ 0,05)"
        )
        kruskal_text = f"Kruskal-Wallis   H = {stat:.4f}   p-valor = {pval:.4f}   →   {conclusao}"
    else:
        kruskal_text = "Grupos insuficientes para o teste."

    return fig, kruskal_text


if __name__ == '__main__':
    app.run(debug=True)