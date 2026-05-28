from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Rótulos de exibição padronizados (sem underline)
TIERS = ['Tier 1', 'Tier 2', 'Tier 3']
LABELS = {'City_Tier_Label': 'Classificação', 'Rent': 'Aluguel ($)'}

layout = html.Div([
    html.H2("5.4 – Custo de Moradia e Saúde Financeira", 
            style={'marginBottom': '20px', 'color': '#1D1252', 'fontSize': '22px'}),
    
    html.Div([
        html.Label("Perspectiva Habitacional:", style={'fontWeight': '600', 'marginRight': '16px', 'display': 'block', 'marginBottom': '8px'}),
        dcc.RadioItems(
            id='s54-view',
            options=[
                {'label': ' Dispersão do Aluguel (Box)',     'value': 'boxplot'},
                {'label': ' Densidade do Aluguel (Violin)',  'value': 'violin'},
                {'label': ' Estatísticas Básicas',           'value': 'stats'},
                {'label': ' Comprometimento da Renda (%)',   'value': 'ratio'},
            ],
            value='ratio', # Começa pela visão mais relevante financeiramente
            style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '12px'},
            inputStyle={'marginRight': '6px'},
            labelStyle={'color': '#333', 'cursor': 'pointer'},
        ),
    ], style={'marginBottom': '24px', 'backgroundColor': '#f8f9fa', 'padding': '16px', 'borderRadius': '8px', 'border': '1px solid #eaeaea'}),

    dcc.Loading(
        type='dot',
        color='#1D1252',
        children=dcc.Graph(id='s54-graph', config={'displayModeBar': False})
    ),

    html.Div([
        html.Strong("Insight Habitacional: ", style={'color': '#1D1252'}),
        html.Span(id='s54-insight')
    ], style={
        'padding': '16px', 
        'backgroundColor': '#f8f9fa', 
        'borderLeft': '4px solid #1D1252',
        'borderRadius': '4px', 
        'marginTop': '16px', 
        'fontSize': '14px'
    }),
])

# ── CALLBACKS DA SEÇÃO 5.4 ─────────────────────────────────────────────────────
def register_callbacks(app):
    @app.callback(
        Output('s54-graph', 'figure'),
        Output('s54-insight', 'children'),
        Input('filtered-data-store', 'data'),
        Input('s54-view', 'value'),
    )
    def update_s54(stored_data, view):
        if not stored_data:
            return {}, "Nenhum dado disponível após os filtros."

        dff = pd.DataFrame(stored_data)
        
        # Trava de Segurança Crítica
        if not all(col in dff.columns for col in ['City_Tier', 'Rent']):
            return {}, html.Span("Erro de Dataset: Colunas 'City_Tier' ou 'Rent' ausentes.", style={'color': '#E54B4B'})

        dff = dff.dropna(subset=['City_Tier', 'Rent']).copy()
        
        # Coluna visual para limpar o underline na exibição
        dff['City_Tier_Label'] = dff['City_Tier'].astype(str).str.replace('_', ' ')

        brand_color = '#1D1252'
        seq_colors = ['#1D1252', '#594CA3', '#A095F8']
        fig = None
        insight = ""

        if view == 'boxplot':
            fig = px.box(dff, x='City_Tier_Label', y='Rent', color='City_Tier_Label',
                         color_discrete_sequence=seq_colors,
                         labels=LABELS, category_orders={'City_Tier_Label': TIERS})
            fig.update_layout(showlegend=False)
            insight = "Avaliando a dispersão. Pontos acima da extremidade superior indicam habitantes pagando alto ágio por localização (outliers habitacionais)."

        elif view == 'violin':
            fig = px.violin(dff, x='City_Tier_Label', y='Rent', color='City_Tier_Label',
                            color_discrete_sequence=seq_colors,
                            box=True, points=False,
                            labels=LABELS, category_orders={'City_Tier_Label': TIERS})
            fig.update_layout(showlegend=False)
            insight = "As áreas mais largas do violino mostram onde a maior massa populacional de cada Tier concentra seus gastos de aluguel."

        elif view == 'stats':
            stats = (dff.groupby('City_Tier_Label')['Rent']
                     .agg(Média='mean', Mediana='median', Desvio='std')
                     .reindex(TIERS).reset_index()
                     .melt(id_vars='City_Tier_Label', var_name='Estatística', value_name='Valor'))
            fig = px.bar(stats, x='City_Tier_Label', y='Valor', color='Estatística',
                         barmode='group', color_discrete_sequence=['#1D1252', '#7B6FCD', '#cccccc'],
                         text=stats['Valor'].apply(lambda v: f'${v:,.0f}'),
                         labels={'City_Tier_Label': 'Classificação', 'Valor': 'Custo ($)'},
                         category_orders={'City_Tier_Label': TIERS})
            fig.update_traces(textposition='outside')
            insight = "Se a Média for notavelmente superior à Mediana, significa que locações de alto luxo estão distorcendo a realidade do custo de vida médio da região."

        else:  # ratio
            if 'Rent_Income_Ratio' not in dff.columns: 
                return {}, "Erro: Coluna 'Rent_Income_Ratio' ausente no dataset."
                
            avg = dff.groupby('City_Tier_Label')['Rent_Income_Ratio'].mean().reindex(TIERS).reset_index()
            fig = px.bar(avg, x='City_Tier_Label', y='Rent_Income_Ratio', color='City_Tier_Label',
                         color_discrete_sequence=seq_colors,
                         text=avg['Rent_Income_Ratio'].apply(lambda v: f'{v:.1%}'),
                         labels={'City_Tier_Label': 'Classificação', 'Rent_Income_Ratio': 'Aluguel / Renda (%)'},
                         category_orders={'City_Tier_Label': TIERS})
            fig.update_traces(textposition='outside', textfont=dict(weight='bold'))
            fig.update_layout(showlegend=False)
            
            # Linha de teto saudável
            fig.add_hline(y=0.30, line_dash='dash', line_color='#E54B4B', line_width=2,
                          annotation_text='Teto Saudável (30%)', annotation_position='top right')
            
            over_limit = avg[avg['Rent_Income_Ratio'] > 0.30]['City_Tier_Label'].tolist()
            if over_limit:
                insight = html.Span([f"Alerta: Residentes do ", html.Strong(", ".join(over_limit)), " ultrapassam a recomendação de 30% da renda, configurando estresse financeiro habitacional."])
            else:
                insight = html.Span("Positivo: Todos os grupos avaliados mantêm, em média, o custo de moradia abaixo do limite prudencial de 30%.", style={'color': '#2e7d32'})

        # Aplicação do Design System no Plotly
        if fig:
            fig.update_layout(
                height=420, 
                plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(t=20, b=40, l=60, r=20),
                font=dict(family="Segoe UI, Roboto, sans-serif", color="#333"),
                title=None, 
            )
            fig.update_xaxes(showline=True, linewidth=1, linecolor='#e0e0e0', showgrid=False)
            fig.update_yaxes(showline=False, showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            
            if view == 'ratio':
                fig.update_yaxes(tickformat=".0%")

        return fig, insight