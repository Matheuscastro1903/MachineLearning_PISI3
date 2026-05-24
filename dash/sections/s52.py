from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import numpy as np

TIER_ORDER = ['Tier_1', 'Tier_2', 'Tier_3']

layout = html.Div([
    html.H2("5.2 – Gasto com Transporte e Acomodação de Estilo de Vida"),

    html.Div([
        html.Label("Visualizar:"),
        dcc.RadioItems(
            id='s52-view',
            options=[
                {'label': ' Tendência Geral',          'value': 'scatter_global'},
                {'label': ' Distribuição por City Tier', 'value': 'boxplot'},
                {'label': ' Dispersão por City Tier',    'value': 'scatter_tier'},
            ],
            value='scatter_global',
            inline=True,
            inputStyle={'marginRight': '4px'},
            labelStyle={'marginRight': '20px'},
        ),
    ], style={'marginBottom': '15px'}),

    dcc.Graph(id='s52-graph'),

    html.Div(id='s52-insight', style={
        'padding': '10px',
        'border': '1px solid #ddd',
        'borderRadius': '4px',
        'marginTop': '8px',
        'fontFamily': 'monospace',
    }),
])


def register_callbacks(app):
    @app.callback(
        Output('s52-graph',   'figure'),
        Output('s52-insight', 'children'),
        Input('filtered-data-store', 'data'),
        Input('s52-view', 'value'),
    )
    def update_s52(stored_data, view):
        if not stored_data:
            return {}, "Nenhum dado disponível."

        dff = pd.DataFrame(stored_data).dropna(subset=['Income', 'Transport', 'City_Tier']).copy()

        if view == 'scatter_global':
            sample = dff.sample(min(2000, len(dff)), random_state=42)
            fig = px.scatter(
                sample, x='Income', y='Transport', opacity=0.35,
                labels={'Income': 'Renda ($)', 'Transport': 'Gasto com Transporte ($)'},
                title='Renda vs. Gasto com Transporte',
            )
            coef = np.polyfit(dff['Income'].to_numpy(dtype=float), dff['Transport'].to_numpy(dtype=float), 1)
            x_line = np.array([dff['Income'].min(), dff['Income'].max()], dtype=float)
            y_line = np.polyval(coef, x_line)
            fig.add_scatter(x=x_line, y=y_line, mode='lines',
                            line=dict(color='red', width=2), name='OLS')
            beta = coef[0]
            insight = (
                f"β = {beta:.5f}   →   "
                f"A cada $1.000 de renda extra, o gasto com transporte cresce ~${beta * 1000:.0f}.   "
                f"Inclinação positiva confirma o fenômeno de Acomodação de Estilo de Vida."
            )

        elif view == 'boxplot':
            fig = px.box(
                dff, x='City_Tier', y='Transport', color='City_Tier',
                labels={'City_Tier': 'Nível da Cidade', 'Transport': 'Gasto com Transporte ($)'},
                title='Distribuição do Gasto com Transporte por City Tier',
                category_orders={'City_Tier': TIER_ORDER},
            )
            fig.update_layout(showlegend=False)
            medians = dff.groupby('City_Tier')['Transport'].median()
            parts = [f"{t}: ${medians[t]:,.0f}" for t in TIER_ORDER if t in medians.index]
            insight = "Medianas por City Tier   →   " + "   |   ".join(parts)

        else:  # scatter_tier
            sample = dff.sample(min(3000, len(dff)), random_state=42)
            fig = px.scatter(
                sample, x='Income', y='Transport', color='City_Tier', opacity=0.35,
                labels={'Income': 'Renda ($)', 'Transport': 'Gasto com Transporte ($)', 'City_Tier': 'City Tier'},
                title='Dispersão: Renda vs. Transporte por City Tier',
                category_orders={'City_Tier': TIER_ORDER},
            )
            betas = {}
            for tier, grp in dff.groupby('City_Tier'):
                if len(grp) >= 2:
                    betas[tier] = np.polyfit(grp['Income'].to_numpy(dtype=float), grp['Transport'].to_numpy(dtype=float), 1)[0]
            parts = [f"{t}: β={betas[t]:.5f}" for t in TIER_ORDER if t in betas]
            insight = (
                "Linhas de tendência por tier praticamente idênticas   →   "
                + "   |   ".join(parts)
                + "   →   O gasto escala com a renda independentemente do City Tier."
            )

        fig.update_layout(height=450, margin=dict(t=50, b=50, l=60, r=20))
        return fig, insight
