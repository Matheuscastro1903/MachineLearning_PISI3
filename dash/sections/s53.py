from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px

# ── PALETA DE CORES (Consistente com a Home) ──
BG_MAIN = "#FFFFFF"
BG_CARD = "#FEFEFF"
TEXT_MAIN = "#000000"
TEXT_MUTED = "#555555"
BORDER_COLOR = "#1D1252"
COLOR_ACCENT = "#1D1252"
COLOR_SUCCESS = "#4ADE80"
COLOR_ERROR = "#F87171"
COLOR_PLOT_FILL = "#7f7f7f"

# ── ESTILO PADRÃO DOS CARDS ──
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

AGE_BINS   = [18, 26, 36, 46, 56, 200]
AGE_LABELS = ['18-25', '26-35', '36-45', '46-55', '55+']
OCC_ORDER  = ['Self Employed', 'Retired', 'Student', 'Professional']
TIER_ORDER = ['Tier 1', 'Tier 2', 'Tier 3']
SEQ_COLORS = ['#1D1252', '#594CA3', '#7B6FCD', '#A095F8', '#CBBFFF']


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

        # ── CABEÇALHO E METODOLOGIA ──
        html.H1("Condicionantes Financeiros ao Longo da Vida", style={'color': TEXT_MAIN, 'fontSize': '36px', 'marginBottom': '16px'}),
        html.P(
            "Esta seção apresenta os resultados da análise exploratória sobre como renda, idade, número de dependentes, "
            "ocupação profissional e nível de urbanização interagem para determinar a capacidade de poupança dos indivíduos.",
            style={'color': TEXT_MUTED, 'fontSize': '18px', 'marginBottom': '40px', 'lineHeight': '1.6'}
        ),

        html.Div([
            html.H3("Metodologia", style={'color': COLOR_ACCENT, 'fontSize': '20px', 'marginBottom': '12px'}),
            html.P(
                "O objetivo da análise exploratória (EDA) é identificar padrões e relações entre variáveis para a formulação de hipóteses "
                "voltadas à previsão de comportamento financeiro e capacidade de poupança. "
                "As análises avançam do mais simples (média de renda por idade) ao mais complexo (ciclo de vida cruzado com dependentes).",
                style=TEXT_P_STYLE
            ),
            html.H4("Principais variáveis analisadas:", style={'color': TEXT_MAIN, 'fontSize': '16px', 'marginBottom': '12px'}),
            html.Ul([
                html.Li([html.Strong("Fixed_Ratio: ", style={'color': COLOR_ACCENT}),
                         "Proporção da renda comprometida com despesas inelásticas (Rent + Loan_Repayment + Insurance)."]),
                html.Li([html.Strong("Disposable_Income: ", style={'color': COLOR_ACCENT}),
                         "Renda líquida restante após todas as despesas — representa o efetivo poder de consumo e poupança."]),
                html.Li([html.Strong("City_Tier: ", style={'color': COLOR_ACCENT}),
                         "Classificação do nível de urbanização da cidade (Tier 1 = maior, Tier 3 = menor)."])
            ], style={'color': TEXT_MUTED, 'lineHeight': '1.8'}),
        ], style=CARD_STYLE),

        # ── 5.3.1 RENDA MÉDIA POR FAIXA ETÁRIA ──
        html.Div([
            html.H2("5.3.1. Renda Média por Faixa Etária", style={'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'}),
            html.P(
                "A primeira análise investiga a relação entre faixa etária e renda média, partindo da hipótese clássica do ciclo de vida "
                "(Modigliani; Brumberg, 1954), segundo a qual a renda tende a crescer com a acumulação de capital humano e experiência "
                "profissional, atingindo um platô nas décadas intermediárias e eventualmente recuando com a aposentadoria.",
                style=TEXT_P_STYLE
            ),

            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-s53-renda-idade-container')),

            criar_bloco_insight(
                "Interpretação: ",
                "Contrariamente à hipótese do ciclo de vida, os resultados revelam uma distribuição notavelmente estável da renda média "
                "entre as faixas etárias. Essa ausência de progressão salarial com a idade sugere que, neste dataset, "
                "a variável Age possui baixo poder preditivo isolado sobre Income."
            )
        ], style=CARD_STYLE),

        # ── 5.3.2 DISTRIBUIÇÃO DE RENDA POR OCUPAÇÃO ──
        html.Div([
            html.H2("5.3.2. Distribuição e Instabilidade de Renda por Ocupação", style={'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'}),
            html.P(
                "A segunda análise avança da média para a distribuição completa da renda dentro de cada categoria ocupacional, "
                "adotando o boxplot como instrumento principal por sua capacidade de revelar simultaneamente a presença de valores "
                "atípicos de renda, que podem distorcer a média como medida de tendência central.",
                style=TEXT_P_STYLE
            ),

            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-s53-renda-ocupacao-container')),

            criar_bloco_insight(
                "Interpretação: ",
                "Todas as categorias ocupacionais apresentam amplitude elevada e cauda superior pronunciada, com valores atípicos "
                "expressivos em Professional e Self Employed. A mediana situa-se consideravelmente abaixo da média em todas as categorias, "
                "confirmando a assimetria positiva — o que limita o poder discriminatório de Occupation como variável isolada."
            )
        ], style=CARD_STYLE),

        # ── 5.3.3 GASTOS FIXOS POR CITY TIER ──
        html.Div([
            html.H2("5.3.3. Comprometimento da Renda com Gastos Fixos por City Tier", style={'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'}),
            html.P(
                "A terceira análise mensurou a pressão orçamentária estrutural imposta pelo contexto urbano por meio do Fixed_Ratio — "
                "proporção da renda total comprometida com aluguel (Rent), serviço de dívida (Loan_Repayment) e seguros (Insurance). "
                "O gráfico apresenta o Fixed_Ratio médio por City_Tier.",
                style=TEXT_P_STYLE
            ),

            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-s53-gastos-fixos-container')),

            criar_bloco_insight(
                "Interpretação: ",
                "Os resultados revelam uma relação decrescente: Tier 1 apresenta Fixed_Ratio de 0,39, ao passo que Tier 2 registra 0,29 "
                "e Tier 3, 0,23. Isso significa que indivíduos em cidades Tier 1 comprometem cerca de 39% da renda bruta com gastos fixos "
                "antes de qualquer despesa variável ou poupança. City_Tier se configura como uma das variáveis com maior potencial "
                "preditivo sobre a saúde financeira real dos indivíduos."
            )
        ], style=CARD_STYLE),

        # ── 5.3.4 MAPA DE CORRELAÇÃO ──
        html.Div([
            html.H2("5.3.4. Mapa de Correlação entre Variáveis Numéricas", style={'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'}),
            html.P(
                "Com o objetivo de identificar, de forma panorâmica, as relações lineares entre todas as variáveis numéricas do conjunto "
                "de dados, foi construído um heatmap de correlação de Pearson. "
                "Cores mais intensas indicam correlações mais fortes (positivas ou negativas).",
                style=TEXT_P_STYLE
            ),

            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-s53-correlacao-container')),

            criar_bloco_insight(
                "Interpretação: ",
                "O mapa evidencia correlações positivas muito fortes entre Income e as variáveis absolutas de gastos (Rent, Loan_Repayment, "
                "Groceries, Transport), o que é esperado — despesas tendem a escalar com a renda. "
                "A correlação negativa entre Dependents e Disposable_Income formaliza o efeito de pressão dos dependentes sobre a renda disponível."
            )
        ], style=CARD_STYLE),

        # ── 5.3.5 RENDA DISPONÍVEL POR FAIXA ETÁRIA E DEPENDENTES ──
        html.Div([
            html.H2("5.3.5. Ciclo de Vida Financeiro: Renda Disponível por Faixa Etária e Dependentes", style={'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'}),
            html.P(
                "Esta análise busca mensurar como o número de dependentes modula a renda disponível ao longo das diferentes fases da vida, "
                "combinando Age_Group e Dependents em uma matriz visualizada por heatmap. "
                "A renda disponível representa o efetivo poder de consumo e poupança do indivíduo após os compromissos fixos.",
                style=TEXT_P_STYLE
            ),

            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-s53-renda-disponivel-container')),

            criar_bloco_insight(
                "Interpretação: ",
                "O padrão mais perceptível é a queda abrupta de renda disponível para indivíduos com pelo menos um dependente, "
                "independentemente da faixa etária: a diferença entre 0 e 1 dependente representa reduções de 19% a 31% na renda disponível média. "
                "A célula de menor renda disponível concentra-se em 36-45 anos com dois dependentes — "
                "fase intermediária da carreira combinada com maior responsabilidade familiar."
            )
        ], style=CARD_STYLE),

        # ── 5.3.6 TRAJETÓRIA DE RENDA POR OCUPAÇÃO ──
        html.Div([
            html.H2("5.3.6. Trajetória de Renda Mediana por Ocupação ao Longo da Vida", style={'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'}),
            html.P(
                "A análise final examinou como as trajetórias de renda ao longo do ciclo de vida diferem entre as categorias ocupacionais, "
                "utilizando a mediana como estatística de centralidade. "
                "O gráfico de linhas com marcadores permite comparar o ritmo de crescimento e o perfil de cada ocupação ao longo das faixas etárias.",
                style=TEXT_P_STYLE
            ),

            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-s53-trajetoria-container')),

            criar_bloco_insight(
                "Interpretação: ",
                "As trajetórias revelam comportamentos estruturalmente distintos entre as ocupações. Student apresenta as menores medianas "
                "em todas as faixas. Retired exibe trajetória estável, condizente com o caráter fixo dos rendimentos da aposentadoria. "
                "Professional e Self Employed apresentam as maiores medianas: Self Employed tende a crescer nas faixas intermediárias "
                "e cai em idades mais avançadas, enquanto Professional mantém crescimento mais regular."
            )
        ], style=CARD_STYLE),

        # ── CONCLUSÃO GERAL ──
        html.Div([
            html.H2("Conclusão Geral", style={'color': COLOR_ACCENT, 'fontSize': '28px', 'marginBottom': '20px'}),
            html.P(
                "Os resultados desta seção indicam que os determinantes da capacidade de poupança neste dataset são predominantemente estruturais:",
                style=TEXT_P_STYLE
            ),

            html.H4("Variáveis de Baixo Poder Preditivo Isolado", style={'color': TEXT_MAIN, 'fontSize': '18px', 'marginBottom': '8px'}),
            html.P(
                "A renda se mostrou praticamente insensível à faixa etária, e a instabilidade intragrupo limita o poder discriminatório "
                "de Occupation como variável isolada. Nenhuma dessas variáveis, por si só, explica diferenças relevantes na saúde financeira.",
                style=TEXT_P_STYLE
            ),

            html.H4("City_Tier como Determinante Estrutural", style={'color': TEXT_MAIN, 'fontSize': '18px', 'marginBottom': '8px'}),
            html.P(
                "O achado de maior impacto prático é o comprometimento diferencial por nível urbano: indivíduos em Tier 1 destinam 39% "
                "da renda a gastos fixos inelásticos contra 23% em Tier 3, resultando em renda disponível média quase duas vezes menor.",
                style=TEXT_P_STYLE
            ),

            criar_bloco_insight(
                "Síntese Analítica: ",
                "City_Tier, Dependents e Age_Group são as variáveis com maior potencial preditivo identificadas nesta seção. "
                "Esses achados, validados pelo mapa de correlação de Pearson, fornecem a base empírica necessária para a construção "
                "da variável-alvo de vulnerabilidade e a seleção das features do modelo preditivo desenvolvido nas seções seguintes."
            )

        ], style=CARD_STYLE)

    ], style={'maxWidth': '1100px', 'margin': '0 auto'})


# ── FUNÇÕES AUXILIARES DE RENDERIZAÇÃO ──

def _pre_processar(data):
    """Computa colunas derivadas necessárias para as análises."""
    if not data:
        return None
    df = pd.DataFrame(data).copy()
    if 'Age_Group_5' not in df.columns and 'Age' in df.columns:
        df['Age_Group_5'] = pd.cut(df['Age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    if 'Fixed_Ratio' not in df.columns:
        cols = ['Rent', 'Loan_Repayment', 'Insurance']
        if all(c in df.columns for c in cols) and 'Income' in df.columns:
            df['Fixed_Ratio'] = (df[cols].sum(axis=1)) / df['Income'].replace(0, 1)
    if 'City_Tier' in df.columns:
        df['City_Tier'] = df['City_Tier'].astype(str).str.replace('_', ' ')
    if 'Occupation' in df.columns:
        df['Occupation'] = df['Occupation'].astype(str).str.replace('_', ' ')
    return df


def _layout_padrao(fig, hovermode="x unified", showlegend=False):
    """Aplica o layout padrão do design system em qualquer figura."""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MAIN, family="Segoe UI, sans-serif"),
        margin=dict(t=50, b=40, l=40, r=20), hovermode=hovermode,
        title_font=dict(size=16, color=COLOR_ACCENT),
        showlegend=showlegend,
        xaxis=dict(showgrid=False, showline=True, linecolor=BORDER_COLOR),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False)
    )
    return fig


def _gerar_bar_renda_idade(df):
    df_plot = df.dropna(subset=['Age_Group_5', 'Income']).copy()
    df_plot['Age_Group_5'] = pd.Categorical(df_plot['Age_Group_5'], categories=AGE_LABELS, ordered=True)
    avg = df_plot.groupby('Age_Group_5', observed=True)['Income'].mean().reset_index()

    fig = px.bar(
        avg, x='Age_Group_5', y='Income',
        text=avg['Income'].apply(lambda v: f'R${v:,.0f}'),
        color_discrete_sequence=[COLOR_PLOT_FILL],
        labels={'Age_Group_5': 'Faixa Etária', 'Income': 'Renda Média (R$)'},
        title='Renda Média por Faixa Etária',
        category_orders={'Age_Group_5': AGE_LABELS}
    )
    fig.update_traces(marker_color=COLOR_ACCENT, textposition='outside', textfont=dict(weight='bold'))
    _layout_padrao(fig)
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def _gerar_box_ocupacao(df):
    df_plot = df.dropna(subset=['Occupation', 'Income']).copy()

    fig = px.box(
        df_plot, x='Occupation', y='Income',
        color='Occupation',
        color_discrete_sequence=SEQ_COLORS,
        labels={'Occupation': 'Ocupação', 'Income': 'Renda (R$)'},
        title='Distribuição de Renda por Ocupação',
        category_orders={'Occupation': OCC_ORDER}
    )
    fig.update_traces(marker=dict(opacity=0.4, size=3))
    _layout_padrao(fig, showlegend=False)
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def _gerar_bar_gastos_fixos(df):
    df_plot = df.dropna(subset=['City_Tier', 'Fixed_Ratio']).copy()
    avg = df_plot.groupby('City_Tier')['Fixed_Ratio'].mean().reindex(TIER_ORDER).reset_index()

    fig = px.bar(
        avg, x='City_Tier', y='Fixed_Ratio',
        text=avg['Fixed_Ratio'].apply(lambda v: f'{v:.1%}' if pd.notna(v) else ''),
        color_discrete_sequence=[COLOR_ACCENT],
        labels={'City_Tier': 'Nível da Cidade', 'Fixed_Ratio': 'Gastos Fixos / Renda'},
        title='Comprometimento Médio da Renda com Gastos Fixos por City Tier',
        category_orders={'City_Tier': TIER_ORDER}
    )
    fig.update_traces(marker_color=COLOR_ACCENT, textposition='outside', textfont=dict(weight='bold'))
    fig.update_yaxes(tickformat='.0%')
    _layout_padrao(fig)
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def _gerar_heatmap_correlacao(df):
    cols_candidatas = [
        'Income', 'Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance',
        'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities',
        'Healthcare', 'Education', 'Miscellaneous',
        'Desired_Savings', 'Disposable_Income', 'Fixed_Ratio'
    ]
    cols = [c for c in cols_candidatas if c in df.columns]
    corr = df[cols].corr(numeric_only=True).round(2)

    fig = px.imshow(
        corr, text_auto=True,
        color_continuous_scale='Purples',
        zmin=-1, zmax=1, aspect='auto',
        title='Mapa de Correlação entre Variáveis Numéricas'
    )
    fig.update_xaxes(tickangle=-45, showgrid=False, showline=False)
    fig.update_yaxes(showgrid=False, showline=False)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MAIN, family="Segoe UI, sans-serif"),
        margin=dict(t=50, b=80, l=80, r=20),
        title_font=dict(size=16, color=COLOR_ACCENT),
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def _gerar_heatmap_renda_disponivel(df):
    df_plot = df.dropna(subset=['Age_Group_5', 'Dependents', 'Disposable_Income']).copy()
    df_plot['Age_Group_5'] = pd.Categorical(df_plot['Age_Group_5'], categories=AGE_LABELS, ordered=True)
    pivot = (
        df_plot.groupby(['Age_Group_5', 'Dependents'], observed=True)['Disposable_Income']
        .mean()
        .unstack('Dependents')
    )
    pivot.index   = pivot.index.astype(str)
    pivot.columns = pivot.columns.astype(str)

    fig = px.imshow(
        pivot, text_auto=True,
        color_continuous_scale='Purples',
        labels={'x': 'Nº de Dependentes', 'y': 'Faixa Etária', 'color': 'Renda Disponível (R$)'},
        aspect='auto',
        title='Renda Disponível Média por Faixa Etária e Número de Dependentes'
    )
    fig.update_xaxes(showgrid=False, showline=False)
    fig.update_yaxes(showgrid=False, showline=False)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MAIN, family="Segoe UI, sans-serif"),
        margin=dict(t=50, b=40, l=80, r=20),
        title_font=dict(size=16, color=COLOR_ACCENT),
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def _gerar_linha_trajetoria(df):
    df_plot = df.dropna(subset=['Age_Group_5', 'Occupation', 'Income']).copy()
    df_plot['Age_Group_5'] = pd.Categorical(df_plot['Age_Group_5'], categories=AGE_LABELS, ordered=True)
    traj = (
        df_plot.groupby(['Age_Group_5', 'Occupation'], observed=True)['Income']
        .median()
        .reset_index()
    )

    fig = px.line(
        traj, x='Age_Group_5', y='Income', color='Occupation',
        markers=True,
        color_discrete_sequence=SEQ_COLORS,
        labels={'Age_Group_5': 'Faixa Etária', 'Income': 'Renda Mediana (R$)', 'Occupation': 'Ocupação'},
        title='Trajetória de Renda Mediana por Ocupação ao Longo da Vida',
        category_orders={'Age_Group_5': AGE_LABELS, 'Occupation': OCC_ORDER}
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=8))
    _layout_padrao(fig, hovermode="x unified", showlegend=True)
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


# ── REGISTO DOS CALLBACKS ──
def register_callbacks(app):

    @app.callback(Output('grafico-s53-renda-idade-container', 'children'), Input('filtered-data-store', 'data'))
    def render_renda_idade(data):
        df = _pre_processar(data)
        if df is None:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        if not all(c in df.columns for c in ['Age_Group_5', 'Income']):
            return html.Div("Colunas ausentes.", style={'color': COLOR_ERROR})
        return _gerar_bar_renda_idade(df)

    @app.callback(Output('grafico-s53-renda-ocupacao-container', 'children'), Input('filtered-data-store', 'data'))
    def render_renda_ocupacao(data):
        df = _pre_processar(data)
        if df is None:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        if not all(c in df.columns for c in ['Occupation', 'Income']):
            return html.Div("Colunas ausentes.", style={'color': COLOR_ERROR})
        return _gerar_box_ocupacao(df)

    @app.callback(Output('grafico-s53-gastos-fixos-container', 'children'), Input('filtered-data-store', 'data'))
    def render_gastos_fixos(data):
        df = _pre_processar(data)
        if df is None:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        if not all(c in df.columns for c in ['City_Tier', 'Fixed_Ratio']):
            return html.Div("Colunas ausentes.", style={'color': COLOR_ERROR})
        return _gerar_bar_gastos_fixos(df)

    @app.callback(Output('grafico-s53-correlacao-container', 'children'), Input('filtered-data-store', 'data'))
    def render_correlacao(data):
        df = _pre_processar(data)
        if df is None:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        return _gerar_heatmap_correlacao(df)

    @app.callback(Output('grafico-s53-renda-disponivel-container', 'children'), Input('filtered-data-store', 'data'))
    def render_renda_disponivel(data):
        df = _pre_processar(data)
        if df is None:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        if not all(c in df.columns for c in ['Age_Group_5', 'Dependents', 'Disposable_Income']):
            return html.Div("Colunas ausentes.", style={'color': COLOR_ERROR})
        return _gerar_heatmap_renda_disponivel(df)

    @app.callback(Output('grafico-s53-trajetoria-container', 'children'), Input('filtered-data-store', 'data'))
    def render_trajetoria(data):
        df = _pre_processar(data)
        if df is None:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        if not all(c in df.columns for c in ['Age_Group_5', 'Occupation', 'Income']):
            return html.Div("Colunas ausentes.", style={'color': COLOR_ERROR})
        return _gerar_linha_trajetoria(df)
