from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from scipy.stats import kruskal


layout = html.Div([
    html.H2("5.1 – Waste Ratio por Variável Demográfica", 
            style={'marginBottom': '20px', 'color': '#1D1252', 'fontSize': '22px'}),
    
    html.Div([
        html.Label("Analisar impacto por:", style={'fontWeight': '600', 'marginRight': '12px'}),
        dcc.RadioItems(
            id='s51-group-var',
            options=[
                {'label': ' Faixa Etária',          'value': 'Age_Group'},
                {'label': ' Ocupação Profissional', 'value': 'Occupation'},
                {'label': ' Nº de Dependentes',     'value': 'Dependents'},
            ],
            value='Age_Group',
            inline=True,
            inputStyle={'marginRight': '6px'},
            labelStyle={'marginRight': '24px', 'color': '#333', 'cursor': 'pointer'},
        ),
    ], style={'marginBottom': '24px', 'display': 'flex', 'alignItems': 'center'}),
    
    dcc.Loading(
        type='dot',
        color='#1D1252',
        children=dcc.Graph(id='s51-violin', config={'displayModeBar': False})
    ),
    
    html.Div([
        html.Strong("Insight Estatístico (Kruskal-Wallis): ", style={'color': '#1D1252'}),
        html.Span(id='s51-kruskal')
    ], style={
        'padding': '16px', 
        'backgroundColor': '#f8f9fa', 
        'borderLeft': '4px solid #1D1252',
        'borderRadius': '4px', 
        'marginTop': '16px', 
        'fontSize': '14px'
    }),
])

def register_callbacks(app):
    @app.callback(
        Output('s51-violin',  'figure'),
        Output('s51-kruskal', 'children'),
        Input('filtered-data-store', 'data'),
        Input('s51-group-var', 'value'),
    )
    def update_s51(stored_data, group_var):
        if not stored_data:
            return {}, "Nenhum dado disponível após os filtros."

        dff = pd.DataFrame(stored_data)
        
        if group_var not in dff.columns or 'Waste_Ratio' not in dff.columns:
            erro_msg = html.Span(f"Erro de Dataset: As colunas '{group_var}' ou 'Waste_Ratio' não foram encontradas.", style={'color': '#E54B4B'})
            return {}, erro_msg

        dff = dff.dropna(subset=[group_var, 'Waste_Ratio']).copy()

        labels = {
            'Age_Group':  'Faixa Etária',
            'Occupation': 'Ocupação Profissional',
            'Dependents': 'Nº de Dependentes',
        }

        # Limpeza visual para os gráficos não exibirem underline (ex: "Jovem_Adulto" vira "Jovem Adulto")
        if group_var in ['Age_Group', 'Occupation']:
            dff[group_var] = dff[group_var].astype(str).str.replace('_', ' ')

        # Ordenação das categorias
        if group_var == 'Age_Group':
            # Nota: Altere a lista abaixo se as categorias de idade do seu banco estiverem em inglês (ex: 'Young Adult')
            order = ['Jovem Adulto', 'Adulto', 'Sênior', 'Idoso'] 
            dff[group_var] = pd.Categorical(dff[group_var], categories=order, ordered=True)
            cat_order = {group_var: order}
        elif group_var == 'Dependents':
            dff[group_var] = dff[group_var].astype(int).astype(str)
            dep_vals = sorted(dff[group_var].unique(), key=int)
            cat_order = {group_var: dep_vals}
        else:
            cat_order = {}

        color_discrete_sequence = ['#1D1252', '#3A2E7A', '#594CA3', '#7B6FCD', '#A095F8']

        fig = px.violin(
            dff.sample(min(300, len(dff)), random_state=42).reset_index(drop=True), x=group_var, y='Waste_Ratio', color=group_var,
            box=True, points='outliers', category_orders=cat_order,
            color_discrete_sequence=color_discrete_sequence,
            labels={'Waste_Ratio': 'Gasto Evitável (%)', group_var: labels.get(group_var, group_var)},
        )
        
   
        fig.update_layout(
            showlegend=False, 
            height=400,
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(t=20, b=40, l=50, r=20),
            font=dict(family="Segoe UI, Roboto, sans-serif", color="#333"),
        )
        fig.update_xaxes(showline=True, linewidth=1, linecolor='#e0e0e0', showgrid=False)
        fig.update_yaxes(title_text='Gasto Evitável (%)', showline=False, showgrid=True, gridwidth=1, gridcolor='#f0f0f0')

        # Cálculo do Kruskal-Wallis com base nos dados filtrados
        groups = [g['Waste_Ratio'].values for _, g in dff.groupby(group_var, observed=True) if len(g) >= 2]
        if len(groups) >= 2:
            stat, pval = kruskal(*groups)
            conclusao = html.Span("Existe diferença significativa de gastos entre os grupos.", style={'color': '#E54B4B', 'fontWeight': 'bold'}) if pval < 0.05 else html.Span("Não há diferença estatística significativa entre os grupos.", style={'color': '#2e7d32', 'fontWeight': 'bold'})
            
            kruskal_text = html.Span([
                f"H = {stat:.2f} | p-valor = {pval:.4f} → ", conclusao
            ])
        else:
            kruskal_text = "Grupos insuficientes para a execução do teste estatístico."

        return fig, kruskal_text