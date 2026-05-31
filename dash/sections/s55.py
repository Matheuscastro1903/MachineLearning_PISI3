from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data import df_master 

min_age_dataset = int(df_master['Age'].min())
max_age_dataset = int(df_master['Age'].max())

layout = html.Div([
    html.H2("5.5 – Gasto com Dopamina e Imposto do Cansaço", 
            style={'marginBottom': '20px', 'color': '#1D1252', 'fontSize': '22px'}),
    
    # ── Painel de Controle ──
    html.Div([
        # Linha 1: Visão da Análise
        html.Div([
            html.Label("Perspectiva Comportamental:", style={'fontWeight': '600', 'marginRight': '16px', 'display': 'block', 'marginBottom': '8px'}),
            dcc.RadioItems(
                id='s55-view',
                options=[
                    {'label': ' 1. Evolução por Idade',          'value': 'idade'},
                    {'label': ' 2. Desvio por Ocupação',         'value': 'ocupacao'},
                    {'label': ' 3. Imposto do Cansaço por Tier', 'value': 'tier'},
                ],
                value='idade',
                style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '10px'},
                inputStyle={'marginRight': '6px'},
                labelStyle={'color': '#333', 'cursor': 'pointer'},
            ),
        ], style={'marginBottom': '20px'}),

        # Linha 2: Checklist de Dopamina
        html.Div([
            html.Label("Composição do 'Gasto com Dopamina':", style={'fontWeight': '600', 'marginRight': '16px', 'display': 'block', 'marginBottom': '8px'}),
            dcc.Checklist(
                id='s55-categories',
                options=[
                    {'label': ' Comer Fora (Eating Out)',   'value': 'Eating_Out'},
                    {'label': ' Entretenimento',            'value': 'Entertainment'},
                    {'label': ' Diversos (Miscellaneous)',  'value': 'Miscellaneous'}
                ],
                value=['Eating_Out', 'Entertainment', 'Miscellaneous'],
                style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'},
                inputStyle={'marginRight': '6px'},
                labelStyle={'color': '#333', 'cursor': 'pointer'},
            )
        ], style={'marginBottom': '24px', 'paddingBottom': '20px', 'borderBottom': '1px solid #eaeaea'}),

        # Linha 3: Slider de Maturidade
        html.Div([
            html.Label("Idade de Maturidade Financeira (Ponto de Inflexão):", style={'fontWeight': '600', 'display': 'block', 'marginBottom': '16px'}),
            dcc.Slider(
                id='s55-maturity-age',
                min=min_age_dataset,
                max=max_age_dataset,
                step=1,
                value=32,
                allow_direct_input=False,
                marks={i: str(i) for i in range(min_age_dataset, max_age_dataset + 1, 5)},
            )
        ])
    ], style={'padding': '24px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'border': '1px solid #eaeaea', 'marginBottom': '24px'}),
    
    dcc.Loading(
        type='dot',
        color='#1D1252',
        children=dcc.Graph(id='s55-graph', config={'displayModeBar': False})
    ),
    
    # ── Card de Insight ──
    html.Div([
        html.Strong("Insight Comportamental: ", style={'color': '#1D1252'}),
        html.Span(id='s55-insight')
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
        Output('s55-graph', 'figure'),
        Output('s55-insight', 'children'),
        Input('filtered-data-store', 'data'),
        Input('s55-view', 'value'),
        Input('s55-categories', 'value'),
        Input('s55-maturity-age', 'value')
    )
    def update_s55(stored_data, view, categories, maturity_age):
        if not stored_data:
            return go.Figure(), "Nenhum dado disponível após os filtros."
            
        if not categories:
            fig = go.Figure().update_layout(
                title="Selecione ao menos uma categoria de gasto para análise.", 
                plot_bgcolor='white', paper_bgcolor='white',
                font=dict(family="Segoe UI", color="#333")
            )
            return fig, "Aguardando seleção de categorias de gasto."

        dff = pd.DataFrame(stored_data)
        
        # Limpeza visual das colunas se existirem
        if 'Occupation' in dff.columns:
            dff['Occupation'] = dff['Occupation'].astype(str).str.replace('_', ' ')
        if 'City_Tier' in dff.columns:
            dff['City_Tier'] = dff['City_Tier'].astype(str).str.replace('_', ' ')
        
        # Trava de Segurança: Garante que as colunas selecionadas existam, senão zera
        for col in categories:
            if col not in dff.columns:
                dff[col] = 0
                
        if 'Income' not in dff.columns or 'Age' not in dff.columns:
            return go.Figure(), html.Span("Erro: Colunas 'Income' ou 'Age' não encontradas no banco de dados.", style={'color': '#E54B4B'})

        dff['Dopamine_Spend'] = dff[categories].sum(axis=1)
        dff['Income_Safe'] = dff['Income'].replace(0, pd.NA) 
        dff['Fatigue_Tax'] = dff['Dopamine_Spend'] / dff['Income_Safe']

        brand_color = '#1D1252'
        highlight_color = '#E54B4B'
        insight = ""

        if view == 'idade':
            df_age = dff.groupby('Age', as_index=False)['Dopamine_Spend'].mean()
            df_age = df_age.sort_values('Age')
            df_age['Tendencia'] = df_age['Dopamine_Spend'].rolling(window=5, min_periods=1, center=True).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_age['Age'], y=df_age['Dopamine_Spend'], 
                                     mode='lines+markers', name='Média Bruta', 
                                     line=dict(color='#d0d0d0', width=1), opacity=0.6))
            fig.add_trace(go.Scatter(x=df_age['Age'], y=df_age['Tendencia'], 
                                     mode='lines', name='Tendência (M. Móvel)', 
                                     line=dict(color=brand_color, width=3)))
            
            fig.add_vline(x=maturity_age, line_dash="dash", line_color=highlight_color, line_width=2)
            fig.add_annotation(x=maturity_age, y=df_age['Dopamine_Spend'].max() * 0.95,
                               text=f"Maturidade ({maturity_age}a)", showarrow=False, xanchor="left", xshift=10, font=dict(color=highlight_color, size=12))

            fig.update_layout(xaxis_title="Idade", yaxis_title="Gasto com Dopamina ($)", hovermode='x unified')
            
            # Cálculo de Insight
            media_antes = df_age[df_age['Age'] < maturity_age]['Dopamine_Spend'].mean()
            media_depois = df_age[df_age['Age'] >= maturity_age]['Dopamine_Spend'].mean()
            if pd.notna(media_antes) and pd.notna(media_depois) and media_antes > 0:
                diff = ((media_depois - media_antes) / media_antes) * 100
                direcao = "queda" if diff < 0 else "aumento"
                insight = html.Span([f"Ajustando o marco de maturidade para {maturity_age} anos, observamos uma ", html.Strong(f"{direcao} de {abs(diff):.1f}%"), " no gasto médio com alívios compensatórios."])

        elif view == 'ocupacao':
            if 'Occupation' not in dff.columns: return go.Figure(), "Coluna 'Occupation' ausente."
            mean_geral = dff['Fatigue_Tax'].mean()
            df_occ = dff.groupby('Occupation', as_index=False)['Fatigue_Tax'].mean()
            df_occ['Desvio_Percentual'] = ((df_occ['Fatigue_Tax'] - mean_geral) / mean_geral) * 100
            df_occ = df_occ.sort_values('Desvio_Percentual')

            # Cores dinâmicas: Azul para abaixo da média, Vermelho para acima
            colors = [brand_color if val < 0 else highlight_color for val in df_occ['Desvio_Percentual']]

            fig = px.bar(df_occ, x='Desvio_Percentual', y='Occupation', orientation='h',
                         labels={'Desvio_Percentual': 'Desvio da Média (%)', 'Occupation': ''})
            
            fig.update_traces(marker_color=colors, texttemplate='%{x:+.1f}%', textposition='outside')
            fig.add_vline(x=0, line_width=2, line_color="#333")
            fig.update_layout(xaxis=dict(automargin=True), yaxis=dict(automargin=True))
            
            if not df_occ.empty:
                max_occ = df_occ.iloc[-1]['Occupation']
                insight = html.Span([html.Strong(f"{max_occ} "), "lidera o 'Imposto do Cansaço', gastando proporcionalmente mais de sua renda para manter o nível basal de dopamina."])

        else: # tier
            if 'City_Tier' not in dff.columns: return go.Figure(), "Coluna 'City_Tier' ausente."
            dff['Fase_de_Vida'] = dff['Age'].apply(lambda x: f'< {maturity_age} anos' if x < maturity_age else f'≥ {maturity_age} anos')
            df_tier = dff.groupby(['City_Tier', 'Fase_de_Vida'], as_index=False)['Fatigue_Tax'].mean()
            
            fig = px.line(df_tier, x='City_Tier', y='Fatigue_Tax', color='Fase_de_Vida', markers=True,
                          color_discrete_sequence=[brand_color, '#A095F8'],
                          labels={'City_Tier': 'Classificação', 'Fatigue_Tax': 'Imposto do Cansaço (% da Renda)', 'Fase_de_Vida': 'Fase'})
            
            fig.update_traces(line=dict(width=3), marker=dict(size=8))
            fig.update_layout(yaxis_tickformat='.1%')
            insight = f"Comparativo do fardo comportamental. Verifica-se como a dinâmica de consumo compensatório varia entre as cidades antes e depois da linha dos {maturity_age} anos."

        # Otimização Global do Layout
        fig.update_layout(
            height=420, 
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(t=20, b=40, l=60, r=20),
            font=dict(family="Segoe UI, Roboto, sans-serif", color="#333"),
            title=None, showlegend=True,
            legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(showline=True, linewidth=1, linecolor='#e0e0e0', showgrid=False)
        fig.update_yaxes(showline=False, showgrid=True, gridwidth=1, gridcolor='#f0f0f0')

        return fig, insight