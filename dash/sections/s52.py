from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import numpy as np

TIER_ORDER = ['Tier_1', 'Tier_2', 'Tier_3']

layout = html.Div([
    html.H2("5.2 – Transporte e Acomodação de Estilo de Vida", 
            style={'marginBottom': '20px', 'color': '#1D1252', 'fontSize': '22px'}),

    html.Div([
        html.Label("Perspectiva de Análise:", style={'fontWeight': '600', 'marginRight': '12px'}),
        dcc.RadioItems(
            id='s52-view',
            options=[
                {'label': ' Tendência Geral (OLS)',       'value': 'scatter_global'},
                {'label': ' Distribuição por City Tier',  'value': 'boxplot'},
                {'label': ' Dispersão por City Tier',     'value': 'scatter_tier'},
            ],
            value='scatter_global',
            inline=True,
            inputStyle={'marginRight': '6px'},
            labelStyle={'marginRight': '24px', 'color': '#333', 'cursor': 'pointer'},
        ),
    ], style={'marginBottom': '24px', 'display': 'flex', 'alignItems': 'center'}),

    dcc.Loading(
        type='dot',
        color='#1D1252',
        children=dcc.Graph(id='s52-graph', config={'displayModeBar': False})
    ),

    html.Div([
        html.Strong("Insight Financeiro: ", style={'color': '#1D1252'}),
        html.Span(id='s52-insight')
    ], style={
        'padding': '16px', 
        'backgroundColor': '#f8f9fa', 
        'borderLeft': '4px solid #1D1252',
        'borderRadius': '4px', 
        'marginTop': '16px', 
        'fontSize': '14px'
    }),
])

# ── CALLBACKS DA SEÇÃO 5.2 ─────────────────────────────────────────────────────
def register_callbacks(app):
    @app.callback(
        Output('s52-graph',   'figure'),
        Output('s52-insight', 'children'),
        Input('filtered-data-store', 'data'),
        Input('s52-view', 'value'),
    )
    def update_s52(stored_data, view):
        if not stored_data:
            return {}, "Nenhum dado disponível após os filtros."

        dff = pd.DataFrame(stored_data)
        
        # Trava de Segurança Crítica
        req_cols = ['Income', 'Transport', 'City_Tier']
        if not all(col in dff.columns for col in req_cols):
            return {}, html.Span(f"Erro: Colunas esperadas {req_cols} não encontradas no dataset.", style={'color': '#E54B4B'})

        dff = dff.dropna(subset=req_cols).copy()
        
        # Cria uma coluna visual limpa apenas para exibição, sem alterar o dado original que será usado para agrupar
        dff['City_Tier_Label'] = dff['City_Tier'].str.replace('_', ' ')
        display_order = [t.replace('_', ' ') for t in TIER_ORDER]

        brand_color = '#1D1252'
        color_sequence = ['#1D1252', '#594CA3', '#A095F8']

        if view == 'scatter_global':
            # Usa o sample original do seu código (até 2000 linhas) para não travar o navegador
            sample = dff.sample(min(2000, len(dff)), random_state=42)
            fig = px.scatter(
                sample, x='Income', y='Transport', 
                opacity=0.4, 
                color_discrete_sequence=[brand_color],
                labels={'Income': 'Renda Mensal ($)', 'Transport': 'Gasto com Transporte ($)'},
            )
            
            # OLS
            coef = np.polyfit(dff['Income'].to_numpy(dtype=float), dff['Transport'].to_numpy(dtype=float), 1)
            x_line = np.array([dff['Income'].min(), dff['Income'].max()], dtype=float)
            y_line = np.polyval(coef, x_line)
            
            fig.add_scatter(x=x_line, y=y_line, mode='lines',
                            line=dict(color='#E54B4B', width=3, dash='dash'), name='Tendência (OLS)')
            
            beta = coef[0]
            insight = html.Span([
                f"Coeficiente Angular (β) = {beta:.4f}. ",
                html.Br(),
                html.Span(f"A cada $1.000 de renda extra, o gasto com transporte cresce em média ${beta * 1000:.0f}. ", style={'fontWeight': 'bold'}),
                "A inclinação positiva confirma o fenômeno de Acomodação de Estilo de Vida no transporte."
            ])

        elif view == 'boxplot':
            fig = px.box(
                dff, x='City_Tier_Label', y='Transport', color='City_Tier_Label',
                color_discrete_sequence=color_sequence,
                labels={'City_Tier_Label': '', 'Transport': 'Gasto com Transporte ($)'},
                category_orders={'City_Tier_Label': display_order},
            )
            medians = dff.groupby('City_Tier_Label')['Transport'].median()
            parts = [f"{t}: ${medians.get(t, 0):,.0f}" for t in display_order]
            insight = html.Span([
                "Medianas de gasto por classificação de cidade: ",
                html.Span(" | ".join(parts), style={'fontWeight': 'bold', 'color': brand_color})
            ])

        else:  # scatter_tier
            sample = dff.sample(min(3000, len(dff)), random_state=42)
            fig = px.scatter(
                sample, x='Income', y='Transport', color='City_Tier_Label', opacity=0.4,
                color_discrete_sequence=color_sequence,
                labels={'Income': 'Renda Mensal ($)', 'Transport': 'Gasto com Transporte ($)', 'City_Tier_Label': 'Classificação'},
                category_orders={'City_Tier_Label': display_order},
            )
            
            betas = {}
            for tier, grp in dff.groupby('City_Tier_Label'):
                if len(grp) >= 2:
                    betas[tier] = np.polyfit(grp['Income'].to_numpy(dtype=float), grp['Transport'].to_numpy(dtype=float), 1)[0]
            
            parts = [f"{t} (β={betas.get(t, 0):.4f})" for t in display_order]
            insight = html.Span([
                "As taxas de aumento são similares: ",
                html.Span(" | ".join(parts), style={'fontWeight': 'bold'}),
                html.Br(),
                html.Span("O custo de transporte escala com a renda independentemente do tamanho da cidade.", style={'color': '#E54B4B', 'fontWeight': '500'})
            ])

        # Design System Layout
        fig.update_layout(
            height=420, 
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(t=30, b=40, l=60, r=20),
            font=dict(family="Segoe UI, Roboto, sans-serif", color="#333"),
            title=None, 
            showlegend=(view == 'scatter_tier' or view == 'scatter_global'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showline=True, linewidth=1, linecolor='#e0e0e0', showgrid=False)
        fig.update_yaxes(showline=False, showgrid=True, gridwidth=1, gridcolor='#f0f0f0')

        return fig, insight