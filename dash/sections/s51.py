from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from scipy.stats import kruskal

layout = html.Div([
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
        'padding': '10px', 'border': '1px solid #ddd',
        'borderRadius': '4px', 'marginTop': '8px', 'fontFamily': 'monospace',
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
            return {}, "Nenhum dado disponível."

        labels = {
            'Age_Group':  'Faixa Etária',
            'Occupation': 'Ocupação Profissional',
            'Dependents': 'Nº de Dependentes',
        }

        dff = pd.DataFrame(stored_data).dropna(subset=[group_var, 'Waste_Ratio']).copy()

        if group_var == 'Age_Group':
            order = ['Jovem Adulto', 'Adulto', 'Sênior', 'Idoso']
            dff[group_var] = pd.Categorical(dff[group_var], categories=order, ordered=True)
            cat_order = {group_var: order}
        elif group_var == 'Dependents':
            dff[group_var] = dff[group_var].astype(int).astype(str)
            dep_vals = sorted(dff[group_var].unique(), key=int)
            cat_order = {group_var: dep_vals}
        else:
            cat_order = {}

        fig = px.violin(
            dff, x=group_var, y='Waste_Ratio', color=group_var,
            box=True, points='outliers', category_orders=cat_order,
            labels={'Waste_Ratio': '% da Renda em Gastos Evitáveis', group_var: labels[group_var]},
        )
        fig.update_layout(
            showlegend=False, height=420,
            margin=dict(t=30, b=50, l=60, r=20),
            yaxis_title='% da Renda em Gastos Evitáveis',
            xaxis_title=labels[group_var],
        )

        groups = [g['Waste_Ratio'].values for _, g in dff.groupby(group_var, observed=True) if len(g) >= 2]
        if len(groups) >= 2:
            stat, pval = kruskal(*groups)
            conclusao = "Diferença significativa (p < 0,05)" if pval < 0.05 else "Sem diferença significativa (p ≥ 0,05)"
            kruskal_text = f"Kruskal-Wallis   H = {stat:.4f}   p-valor = {pval:.4f}   →   {conclusao}"
        else:
            kruskal_text = "Grupos insuficientes para o teste."

        return fig, kruskal_text
