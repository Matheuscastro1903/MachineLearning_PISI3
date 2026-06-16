from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px


BG_MAIN = "#FFFFFF"
BG_CARD = "#FEFEFF"
TEXT_MAIN = "#000000"
TEXT_MUTED = "#555555"
BORDER_COLOR = "#1D1252"
COLOR_ACCENT = "#1D1252"
COLOR_SUCCESS = "#4ADE80"
COLOR_ERROR = "#F87171"

TIERS = ['Tier 1', 'Tier 2', 'Tier 3']
SEQ_COLORS = ['#1D1252', '#594CA3', '#A095F8']


CARD_STYLE = {
    'backgroundColor': BG_CARD,
    'borderRadius': '12px',
    'padding': '32px',
    'marginBottom': '40px',
    'border': '1px solid #E5E7EB',
    'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.04)'
}

TEXT_P_STYLE = {
    'color': TEXT_MUTED,
    'marginBottom': '24px',
    'lineHeight': '1.7',
    'fontSize': '16px',
    'textAlign': 'justify'
}


def criar_bloco_insight(texto_negrito, texto_normal):
    """Função auxiliar para gerar caixas de insight estilizadas."""
    return html.Div([
        html.Strong(texto_negrito, style={'color': COLOR_ACCENT}),
        html.Span(texto_normal, style={'color': TEXT_MAIN, 'lineHeight': '1.6'})
    ], style={
        'backgroundColor': 'rgba(29, 18, 82, 0.05)',
        'padding': '20px',
        'borderRadius': '8px',
        'borderLeft': f'4px solid {COLOR_ACCENT}',
        'marginTop': '24px'
    })


def get_layout():
    """Retorna o layout estrutural da página. Os gráficos são injetados vazios."""
    return html.Div([

        html.H1("5.4 – Custo de Moradia e Saúde Financeira", style={
            'color': TEXT_MAIN, 'fontSize': '36px', 'marginBottom': '16px'
        }),
        html.P(
            "Esta seção investiga como o custo de moradia varia entre as classificações de cidade (City Tier) "
            "e qual o impacto desse custo sobre a saúde financeira dos indivíduos. "
            "A moradia costuma ser a maior despesa fixa de uma família, e seu peso relativo na renda é "
            "um dos principais indicadores de estresse financeiro.",
            style={'color': TEXT_MUTED, 'fontSize': '18px', 'marginBottom': '40px', 'lineHeight': '1.6'}
        ),

        html.Div([
            html.H3("Metodologia", style={'color': COLOR_ACCENT, 'fontSize': '20px', 'marginBottom': '12px'}),
            html.P(
                "Analisaremos três dimensões do custo habitacional: a dispersão e a densidade do aluguel por tier, "
                "as estatísticas descritivas de cada grupo e, por fim, o comprometimento percentual da renda "
                "com moradia — métrica central para avaliar a sustentabilidade financeira.",
                style=TEXT_P_STYLE
            ),
            html.H4("Referência: A Regra dos 30%", style={'color': TEXT_MAIN, 'fontSize': '16px', 'marginBottom': '12px'}),
            html.Ul([
                html.Li([html.Strong("Abaixo de 30%: ", style={'color': COLOR_SUCCESS}),
                         "A moradia está dentro de uma faixa sustentável, deixando renda disponível para outros gastos e poupança."]),
                html.Li([html.Strong("Acima de 30%: ", style={'color': COLOR_ERROR}),
                         "Estresse financeiro habitacional. O custo de moradia compromete a saúde do orçamento restante."])
            ], style={'color': TEXT_MUTED, 'lineHeight': '1.8'}),
        ], style=CARD_STYLE),

        #  5.4.1 DISPERSÃO DO ALUGUEL (BOXPLOT) 
        html.Div([
            html.H2("5.4.1. Dispersão do Aluguel por City Tier", style={
                'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'
            }),
            html.P(
                "O boxplot abaixo revela a amplitude dos valores de aluguel em cada tier. "
                "Pontos acima da extremidade superior indicam habitantes que pagam um ágio significativo "
                "por localização — os chamados outliers habitacionais.",
                style=TEXT_P_STYLE
            ),
            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-s54-boxplot-container')),
            criar_bloco_insight(
                "Leitura do Boxplot: ",
                "Quanto maior a caixa e mais espalhados os bigodes, maior a heterogeneidade no custo de moradia "
                "dentro daquele tier — indicando que viver numa mesma classificação de cidade pode representar "
                "realidades financeiras muito distintas."
            )
        ], style=CARD_STYLE),

        #  5.4.2 DENSIDADE DO ALUGUEL (VIOLIN) 
        html.Div([
            html.H2("5.4.2. Densidade do Aluguel por City Tier", style={
                'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'
            }),
            html.P(
                "O gráfico de violino complementa o boxplot ao mostrar onde a maior massa de pessoas "
                "de cada tier concentra seus gastos com aluguel. As regiões mais largas do violino "
                "correspondem aos valores mais frequentes na distribuição.",
                style=TEXT_P_STYLE
            ),
            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-s54-violin-container')),
            criar_bloco_insight(
                "Densidade Habitacional: ",
                "Se os violinos de diferentes tiers apresentam formas semelhantes, isso sugere que a "
                "classificação da cidade tem pouca influência sobre o comportamento de gasto com moradia — "
                "padrão relevante para o modelo de vulnerabilidade financeira."
            )
        ], style=CARD_STYLE),

        #  5.4.3 COMPROMETIMENTO DA RENDA
        html.Div([
            html.H2("5.4.3. Comprometimento da Renda com Moradia", style={
                'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'
            }),
            html.P(
                "Esta é a métrica mais relevante desta seção. O percentual de comprometimento da renda "
                "com aluguel (Rent / Income) indica se a moradia está gerando estresse financeiro. "
                "A linha vermelha tracejada marca o limite prudencial de 30% recomendado.",
                style=TEXT_P_STYLE
            ),
            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-s54-ratio-container')),
            criar_bloco_insight(
                "Teto de 30%: ",
                "Grupos que ultrapassam esse limite em média estão, coletivamente, em situação de "
                "estresse habitacional — condição que reduz a capacidade de poupança e aumenta "
                "a vulnerabilidade financeira a imprevistos."
            )
        ], style=CARD_STYLE),

        #  5.4.4 ESTATÍSTICAS DESCRITIVAS 
        html.Div([
            html.H2("5.4.4. Estatísticas Descritivas do Aluguel por Tier", style={
                'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'
            }),
            html.P(
                "O gráfico de barras agrupadas compara média, mediana e desvio padrão do aluguel "
                "em cada tier. A diferença entre média e mediana é especialmente reveladora: "
                "quando a média supera significativamente a mediana, locações de alto valor estão "
                "distorcendo a percepção do custo de vida médio da região.",
                style=TEXT_P_STYLE
            ),
            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-s54-stats-container')),
            criar_bloco_insight(
                "Média vs. Mediana: ",
                "Uma média notavelmente superior à mediana indica presença de outliers de alto valor "
                "puxando a percepção do custo habitacional para cima, enquanto a maioria das pessoas "
                "paga valores bem mais modestos."
            )
        ], style=CARD_STYLE),

       
        html.Div([
            html.H2("Conclusão Geral da Análise Habitacional", style={
                'color': COLOR_ACCENT, 'fontSize': '28px', 'marginBottom': '20px'
            }),
            html.P(
                "A análise do custo de moradia revela o peso estrutural da habitação no orçamento familiar "
                "e sua relação direta com a saúde financeira dos indivíduos:",
                style=TEXT_P_STYLE
            ),

            html.H4("Variação entre Tiers", style={'color': TEXT_MAIN, 'fontSize': '18px', 'marginBottom': '8px'}),
            html.P(
                "As cidades de Tier 1 tendem a apresentar maiores custos absolutos de aluguel, "
                "refletindo o maior custo de vida nos grandes centros. No entanto, o comprometimento "
                "relativo da renda pode não seguir o mesmo padrão, dependendo dos níveis salariais de cada grupo.",
                style=TEXT_P_STYLE
            ),

            html.H4("Implicação para o Modelo de Vulnerabilidade", style={'color': TEXT_MAIN, 'fontSize': '18px', 'marginBottom': '8px'}),
            html.P(
                "O percentual de comprometimento da renda com moradia é uma variável candidata "
                "relevante para o modelo M1. Indivíduos que já destinam uma fatia desproporcional "
                "da renda à habitação possuem menor margem para absorver outros gastos e imprevistos.",
                style=TEXT_P_STYLE
            ),

            html.Div([
                html.Strong("Análise Social: ", style={'color': COLOR_ACCENT}),
                html.Span(
                    "O custo de moradia é um fator estrutural que transcende escolhas individuais. "
                    "Políticas de habitação acessível e planejamento urbano têm impacto direto na "
                    "capacidade das famílias de atingir equilíbrio financeiro.",
                    style={'color': TEXT_MAIN, 'lineHeight': '1.6'}
                )
            ], style={
                'backgroundColor': 'rgba(29, 18, 82, 0.05)',
                'padding': '20px', 'borderRadius': '8px',
                'borderLeft': f'4px solid {COLOR_ACCENT}'
            })
        ], style=CARD_STYLE)

    ], style={'maxWidth': '1100px', 'margin': '0 auto'})




def _gerar_boxplot(df):
    """Gera boxplot do aluguel por City Tier."""
    df_clean = df.dropna(subset=['City_Tier', 'Rent']).copy()
    df_clean['City_Tier_Label'] = df_clean['City_Tier'].astype(str).str.replace('_', ' ')

    fig = px.box(
        df_clean.sample(min(300, len(df_clean)), random_state=42).reset_index(drop=True), x='City_Tier_Label', y='Rent',
        color='City_Tier_Label',
        color_discrete_sequence=SEQ_COLORS,
        category_orders={'City_Tier_Label': TIERS},
        labels={'City_Tier_Label': 'Nível da Cidade', 'Rent': 'Aluguel ($)'},
        title='Dispersão do Aluguel por City Tier'
    )
    fig.update_traces(marker=dict(opacity=0.4, size=3))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MAIN, family="Segoe UI, sans-serif"),
        margin=dict(t=50, b=40, l=40, r=20), hovermode="x unified",
        title_font=dict(size=16, color=COLOR_ACCENT),
        showlegend=False,
        xaxis=dict(showgrid=False, showline=True, linecolor=BORDER_COLOR),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False)
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def _gerar_violin(df):
    """Gera violin do aluguel por City Tier."""
    df_clean = df.dropna(subset=['City_Tier', 'Rent']).copy()
    df_clean['City_Tier_Label'] = df_clean['City_Tier'].astype(str).str.replace('_', ' ')

    fig = px.violin(
        df_clean.sample(min(300, len(df_clean)), random_state=42).reset_index(drop=True), x='City_Tier_Label', y='Rent',
        color='City_Tier_Label',
        color_discrete_sequence=SEQ_COLORS,
        box=True, points=False,
        category_orders={'City_Tier_Label': TIERS},
        labels={'City_Tier_Label': 'Nível da Cidade', 'Rent': 'Aluguel ($)'},
        title='Densidade do Aluguel por City Tier'
    )
    fig.update_traces(opacity=0.7, line_width=1.5, line_color='black', meanline_visible=True, meanline_width=2)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MAIN, family="Segoe UI, sans-serif"),
        margin=dict(t=50, b=40, l=40, r=20), hovermode="x unified",
        title_font=dict(size=16, color=COLOR_ACCENT),
        showlegend=False,
        xaxis=dict(showgrid=False, showline=True, linecolor=BORDER_COLOR),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False)
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def _gerar_ratio(df):
    """Gera bar chart do comprometimento da renda com aluguel por City Tier."""
    df_clean = df.dropna(subset=['City_Tier', 'Rent', 'Income']).copy()
    df_clean['City_Tier_Label'] = df_clean['City_Tier'].astype(str).str.replace('_', ' ')
    df_clean['Rent_Income_Ratio'] = df_clean['Rent'] / df_clean['Income'].replace(0, 1)

    avg = (
        df_clean.groupby('City_Tier_Label')['Rent_Income_Ratio']
        .mean()
        .reindex(TIERS)
        .reset_index()
    )
    avg.columns = ['City_Tier_Label', 'Rent_Income_Ratio']

    fig = px.bar(
        avg, x='City_Tier_Label', y='Rent_Income_Ratio',
        color='City_Tier_Label',
        color_discrete_sequence=SEQ_COLORS,
        text=avg['Rent_Income_Ratio'].apply(lambda v: f'{v:.1%}'),
        category_orders={'City_Tier_Label': TIERS},
        labels={'City_Tier_Label': 'Nível da Cidade', 'Rent_Income_Ratio': 'Aluguel / Renda (%)'},
        title='Comprometimento Médio da Renda com Moradia por City Tier'
    )
    fig.update_traces(textposition='outside', textfont=dict(color=TEXT_MAIN))
    fig.add_hline(
        y=0.30, line_dash='dash', line_color=COLOR_ERROR, line_width=2,
        annotation_text='Teto Saudável (30%)', annotation_position='top right'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MAIN, family="Segoe UI, sans-serif"),
        margin=dict(t=50, b=40, l=40, r=20), hovermode="x unified",
        title_font=dict(size=16, color=COLOR_ACCENT),
        showlegend=False,
        xaxis=dict(showgrid=False, showline=True, linecolor=BORDER_COLOR),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False, tickformat='.0%')
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def _gerar_stats(df):
    """Gera bar chart agrupado com média, mediana e desvio padrão do aluguel por tier."""
    df_clean = df.dropna(subset=['City_Tier', 'Rent']).copy()
    df_clean['City_Tier_Label'] = df_clean['City_Tier'].astype(str).str.replace('_', ' ')

    stats = (
        df_clean.groupby('City_Tier_Label')['Rent']
        .agg(Média='mean', Mediana='median', Desvio='std')
        .reindex(TIERS)
        .reset_index()
        .melt(id_vars='City_Tier_Label', var_name='Estatística', value_name='Valor')
    )
    fig = px.bar(
        stats, x='City_Tier_Label', y='Valor', color='Estatística',
        barmode='group',
        color_discrete_sequence=['#1D1252', '#7B6FCD', '#cccccc'],
        text=stats['Valor'].apply(lambda v: f'${v:,.0f}'),
        category_orders={'City_Tier_Label': TIERS},
        labels={'City_Tier_Label': 'Nível da Cidade', 'Valor': 'Custo ($)'},
        title='Estatísticas Descritivas do Aluguel por City Tier'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MAIN, family="Segoe UI, sans-serif"),
        margin=dict(t=50, b=40, l=40, r=20), hovermode="x unified",
        title_font=dict(size=16, color=COLOR_ACCENT),
        showlegend=True,
        xaxis=dict(showgrid=False, showline=True, linecolor=BORDER_COLOR),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False)
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})



def register_callbacks(app):

    @app.callback(Output('grafico-s54-boxplot-container', 'children'), Input('filtered-data-store', 'data'))
    def render_boxplot(data):
        if not data:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        df = pd.DataFrame(data)
        if not all(c in df.columns for c in ['City_Tier', 'Rent']):
            return html.Div("Colunas ausentes.", style={'color': COLOR_ERROR})
        return _gerar_boxplot(df)

    @app.callback(Output('grafico-s54-violin-container', 'children'), Input('filtered-data-store', 'data'))
    def render_violin(data):
        if not data:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        df = pd.DataFrame(data)
        if not all(c in df.columns for c in ['City_Tier', 'Rent']):
            return html.Div("Colunas ausentes.", style={'color': COLOR_ERROR})
        return _gerar_violin(df)

    @app.callback(Output('grafico-s54-ratio-container', 'children'), Input('filtered-data-store', 'data'))
    def render_ratio(data):
        if not data:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        df = pd.DataFrame(data)
        if not all(c in df.columns for c in ['City_Tier', 'Rent', 'Income']):
            return html.Div("Colunas ausentes.", style={'color': COLOR_ERROR})
        return _gerar_ratio(df)

    @app.callback(Output('grafico-s54-stats-container', 'children'), Input('filtered-data-store', 'data'))
    def render_stats(data):
        if not data:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        df = pd.DataFrame(data)
        if not all(c in df.columns for c in ['City_Tier', 'Rent']):
            return html.Div("Colunas ausentes.", style={'color': COLOR_ERROR})
        return _gerar_stats(df)
