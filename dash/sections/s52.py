from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import numpy as np

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
        html.H1("Análise Exploratória do Gasto com Transporte", style={'color': TEXT_MAIN, 'fontSize': '36px', 'marginBottom': '16px'}),
        html.P(
            "Esta seção investiga uma das crenças mais comuns no planejamento financeiro: a de que o custo de transporte "
            "é determinado quase exclusivamente pela infraestrutura da cidade onde se vive (City Tier).",
            style={'color': TEXT_MUTED, 'fontSize': '18px', 'marginBottom': '40px', 'lineHeight': '1.6'}
        ),

        html.Div([
            html.H3("Metodologia", style={'color': COLOR_ACCENT, 'fontSize': '20px', 'marginBottom': '12px'}),
            html.P(
                "A análise é conduzida em duas etapas. Primeiro, investigamos o comportamento base do gasto com transporte em relação "
                "à renda total para toda a base de dados. Em seguida, segmentamos os dados por City Tier para verificar se a localização "
                "geográfica é, de fato, o principal fator nos custos de mobilidade.",
                style=TEXT_P_STYLE
            ),
            html.H4("O que é a 'Acomodação de Estilo de Vida'?", style={'color': TEXT_MAIN, 'fontSize': '16px', 'marginBottom': '12px'}),
            html.Ul([
                html.Li([html.Strong("Comportamento Elástico: ", style={'color': COLOR_ERROR}),
                         "À medida que a renda cresce, há uma tendência de migração do transporte público (baixo custo) para veículos próprios ou aplicativos (alto custo)."]),
                html.Li([html.Strong("Inclinação Positiva (Beta > 0): ", style={'color': COLOR_ACCENT}),
                         "Confirma estatisticamente que o gasto com transporte não é fixo — ele escala proporcionalmente ao aumento do poder aquisitivo."])
            ], style={'color': TEXT_MUTED, 'lineHeight': '1.8'}),
        ], style=CARD_STYLE),

        # ── 5.2.1 ACOMODAÇÃO DE ESTILO DE VIDA ──
        html.Div([
            html.H2("5.2.1. O Fenômeno da Acomodação de Estilo de Vida", style={'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'}),
            html.P(
                "Antes de analisar a geografia, é necessário compreender o comportamento de consumo básico. "
                "O gráfico abaixo apresenta a relação entre a renda mensal e o gasto com transporte para toda a base de dados. "
                "A linha vermelha mostra o modelo matemático de Regressão Linear OLS (Ordinary Least Squares): "
                "a inclinação positiva da reta confirma estatisticamente que o transporte se comporta como um gasto elástico.",
                style=TEXT_P_STYLE
            ),

            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-transporte-ols-container')),

            criar_bloco_insight(
                "Interpretação: ",
                "A inclinação positiva da reta OLS confirma que o gasto com transporte não é fixo — ele escala proporcionalmente "
                "ao aumento do poder aquisitivo. O progresso financeiro atua como um gatilho para o 'upgrade' no estilo de vida, "
                "onde a escolha por modalidades privadas de transporte é uma decisão de consumo, e não apenas uma imposição geográfica."
            )
        ], style=CARD_STYLE),

        # ── 5.2.2 DISTRIBUIÇÃO POR CITY TIER (BOXPLOT) ──
        html.Div([
            html.H2("5.2.2. Análise Comparativa por City Tier", style={'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'}),
            html.P(
                "A fim de confrontar o senso comum de que a localização geográfica (CEP) é o principal determinante dos custos, "
                "os dados foram segmentados por níveis de cidade (Tiers 1, 2 e 3). "
                "O boxplot abaixo compara a distribuição central e a dispersão dos gastos com transporte entre os três grupos.",
                style=TEXT_P_STYLE
            ),

            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-transporte-boxplot-container')),

            criar_bloco_insight(
                "Interpretação: ",
                "Apesar da distinção entre os três níveis de cidades, o comportamento dos gastos não apresenta a variação drástica "
                "comumente associada ao estereótipo de contraste entre centros desenvolvidos e subdesenvolvidos. "
                "O boxplot revela uma estabilidade estatística entre os grupos — sugerindo que outros fatores têm maior peso do que o City Tier isolado."
            )
        ], style=CARD_STYLE),

        # ── 5.2.2 TENDÊNCIA POR CITY TIER (DISPERSÃO) ──
        html.Div([
            html.H2("5.2.2. Dispersão e Tendência por City Tier", style={'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'}),
            html.P(
                "O gráfico de dispersão a seguir exibe as linhas de tendência OLS para cada City Tier sobrepostas. "
                "Essa visualização é o passo final para verificar se o padrão de elasticidade do transporte em relação à renda "
                "se repete de forma uniforme entre os diferentes níveis de urbanização.",
                style=TEXT_P_STYLE
            ),

            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-transporte-tier-container')),

            criar_bloco_insight(
                "Interpretação: ",
                "As três linhas de tendência (OLS) para cada Tier apresentam uma sobreposição quase absoluta. "
                "Isso demonstra que o gasto em relação à renda é uma constante comportamental que não depende diretamente "
                "da infraestrutura urbana disponível. O CEP, portanto, não é o principal determinante dos custos de mobilidade: "
                "a decisão de consumo por conveniência é."
            )
        ], style=CARD_STYLE),

        # ── CONCLUSÃO GERAL ──
        html.Div([
            html.H2("Conclusão Geral da Análise de Transporte", style={'color': COLOR_ACCENT, 'fontSize': '28px', 'marginBottom': '20px'}),
            html.P(
                "A partir das análises conduzidas, consolidamos os seguintes achados sobre o comportamento do gasto com transporte:",
                style=TEXT_P_STYLE
            ),

            html.H4("O Transporte como Gasto Elástico", style={'color': TEXT_MAIN, 'fontSize': '18px', 'marginBottom': '8px'}),
            html.P(
                "A regressão OLS confirmou que o gasto com transporte cresce proporcionalmente à renda. Indivíduos de maior poder "
                "aquisitivo tendem a optar por modalidades privadas mais onerosas — uma decisão comportamental, não uma necessidade estrutural.",
                style=TEXT_P_STYLE
            ),

            html.H4("City Tier: Influência Menor do que o Esperado", style={'color': TEXT_MAIN, 'fontSize': '18px', 'marginBottom': '8px'}),
            html.P(
                "A segmentação por City Tier revelou que, apesar das diferenças absolutas nos custos de vida, "
                "o padrão de elasticidade do transporte em relação à renda é praticamente idêntico entre os três níveis de cidade. "
                "As linhas de tendência OLS se sobrepõem quase completamente.",
                style=TEXT_P_STYLE
            ),

            criar_bloco_insight(
                "Síntese Analítica: ",
                "O gasto com transporte não depende diretamente apenas do nível de desenvolvimento da cidade. "
                "Uma possível causa desse comportamento são fatores sociológicos pela busca da autonomia. "
                "O progresso financeiro atua como um gatilho para o 'upgrade' no estilo de vida, onde a escolha por "
                "modalidades privadas de transporte é uma decisão de consumo — e não apenas uma imposição geográfica."
            )

        ], style=CARD_STYLE)

    ], style={'maxWidth': '1100px', 'margin': '0 auto'})


# ── FUNÇÕES AUXILIARES DE RENDERIZAÇÃO ──

def _gerar_scatter_ols(df):
    """Gera gráfico de dispersão com linha de tendência OLS (renda vs transporte)."""
    df_clean = df.dropna(subset=['Income', 'Transport']).copy()
    sample = df_clean.sample(min(2000, len(df_clean)), random_state=42)

    fig = px.scatter(
        sample, x='Income', y='Transport',
        color_discrete_sequence=[COLOR_PLOT_FILL],
        labels={'Income': 'Renda Mensal ($)', 'Transport': 'Custo de Transporte ($)'},
        title='Crescimento de Tendência de Gastos com Transporte'
    )

    fig.update_traces(marker=dict(opacity=0.3, size=4))

    coef = np.polyfit(df_clean['Income'].to_numpy(dtype=float), df_clean['Transport'].to_numpy(dtype=float), 1)
    x_line = np.array([df_clean['Income'].min(), df_clean['Income'].max()], dtype=float)
    y_line = np.polyval(coef, x_line)
    fig.add_scatter(x=x_line, y=y_line, mode='lines',
                    line=dict(color='red', width=2), name='Tendência OLS', showlegend=False)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MAIN, family="Segoe UI, sans-serif"),
        margin=dict(t=50, b=40, l=40, r=20), hovermode="closest",
        title_font=dict(size=16, color=COLOR_ACCENT),
        showlegend=False,
        xaxis=dict(showgrid=False, showline=True, linecolor=BORDER_COLOR),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False)
    )

    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def _gerar_boxplot_tier(df):
    """Gera boxplot do gasto com transporte por City Tier."""
    df_clean = df.dropna(subset=['City_Tier', 'Transport']).copy()
    df_clean['City_Tier_Label'] = df_clean['City_Tier'].astype(str).str.replace('_', ' ')
    ordem = ['Tier 1', 'Tier 2', 'Tier 3']

    fig = px.box(
        df_clean, x='City_Tier_Label', y='Transport',
        color='City_Tier_Label',
        color_discrete_sequence=['#1D1252', '#594CA3', '#A095F8'],
        category_orders={'City_Tier_Label': ordem},
        labels={'City_Tier_Label': 'Nível da Cidade', 'Transport': 'Custo de Transporte ($)'},
        title='Comparação de Gastos com Transporte segmentados por City Tier'
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


def _gerar_scatter_tier(df):
    """Gera gráfico de dispersão com linhas OLS por City Tier."""
    df_clean = df.dropna(subset=['City_Tier', 'Income', 'Transport']).copy()
    df_clean['City_Tier_Label'] = df_clean['City_Tier'].astype(str).str.replace('_', ' ')
    ordem = ['Tier 1', 'Tier 2', 'Tier 3']
    cores = ['#1D1252', '#594CA3', '#A095F8']
    sample = df_clean.sample(min(3000, len(df_clean)), random_state=42)

    fig = px.scatter(
        sample, x='Income', y='Transport',
        color='City_Tier_Label',
        color_discrete_sequence=cores,
        category_orders={'City_Tier_Label': ordem},
        labels={'Income': 'Renda ($)', 'Transport': 'Gasto com Transporte ($)', 'City_Tier_Label': 'City Tier'},
        title='Análise de Dispersão e Tendência de Gastos com Transporte entre as Cidades'
    )

    fig.update_traces(marker=dict(opacity=0.2, size=3))

    for tier, cor in zip(ordem, cores):
        grp = df_clean[df_clean['City_Tier_Label'] == tier]
        if len(grp) >= 2:
            coef = np.polyfit(grp['Income'].to_numpy(dtype=float), grp['Transport'].to_numpy(dtype=float), 1)
            x_line = np.array([grp['Income'].min(), grp['Income'].max()], dtype=float)
            y_line = np.polyval(coef, x_line)
            fig.add_scatter(x=x_line, y=y_line, mode='lines',
                            line=dict(color=cor, width=2), name=f'OLS {tier}', showlegend=False)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MAIN, family="Segoe UI, sans-serif"),
        margin=dict(t=50, b=40, l=40, r=20), hovermode="closest",
        title_font=dict(size=16, color=COLOR_ACCENT),
        xaxis=dict(showgrid=False, showline=True, linecolor=BORDER_COLOR),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False)
    )

    return dcc.Graph(figure=fig, config={'displayModeBar': False})


# ── REGISTO DOS CALLBACKS ──
def register_callbacks(app):

    @app.callback(Output('grafico-transporte-ols-container', 'children'), Input('filtered-data-store', 'data'))
    def render_scatter_ols(data):
        if not data:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        df = pd.DataFrame(data)
        if not all(c in df.columns for c in ['Income', 'Transport']):
            return html.Div("Colunas ausentes.", style={'color': COLOR_ERROR})
        return _gerar_scatter_ols(df)

    @app.callback(Output('grafico-transporte-boxplot-container', 'children'), Input('filtered-data-store', 'data'))
    def render_boxplot(data):
        if not data:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        df = pd.DataFrame(data)
        if not all(c in df.columns for c in ['City_Tier', 'Transport']):
            return html.Div("Colunas ausentes.", style={'color': COLOR_ERROR})
        return _gerar_boxplot_tier(df)

    @app.callback(Output('grafico-transporte-tier-container', 'children'), Input('filtered-data-store', 'data'))
    def render_scatter_tier(data):
        if not data:
            return html.Div("Sem dados.", style={'color': COLOR_ERROR})
        df = pd.DataFrame(data)
        if not all(c in df.columns for c in ['City_Tier', 'Income', 'Transport']):
            return html.Div("Colunas ausentes.", style={'color': COLOR_ERROR})
        return _gerar_scatter_tier(df)
