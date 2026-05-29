from dash import html

# ── PALETA DE CORES (Branco e Verde Flat) ──
COLOR_BG = "#FFFFFF"
COLOR_SURFACE = "#F9FAFB" # Cinza super claro para fundos secundários
COLOR_PRIMARY = "#1D1252" # Verde Principal

COLOR_TEXT_MAIN = "#1F2937"
COLOR_TEXT_MUTED = "#6B7280"
COLOR_BORDER = "#E5E7EB"

# ── COMPONENTES REUTILIZÁVEIS (Micro-Design System) ──
def card_metrica(titulo, media, std, min_val=None, max_val=None):
    """Cria um card padronizado para as métricas numéricas"""
    detalhes = [
        html.Div([html.Strong("Média: "), f"{media}"]),
        html.Div([html.Strong("Desvio Padrão (std): "), f"{std}"])
    ]
    if min_val and max_val:
        detalhes.append(html.Div([html.Strong("Mín/Máx: "), f"{min_val} a {max_val}"]))
        
    return html.Div([
        html.H5(titulo, style={'margin': '0 0 10px 0', 'color': COLOR_PRIMARY, 'fontSize': '16px'}),
        html.Div(detalhes, style={'fontSize': '14px', 'color': COLOR_TEXT_MAIN, 'lineHeight': '1.6'})
    ], style={
        'backgroundColor': COLOR_BG, 'padding': '16px', 'borderRadius': '6px',
        'border': f'1px solid {COLOR_BORDER}', 'borderLeft': f'4px solid {COLOR_PRIMARY}',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.05)'
    })

def card_categorico(titulo, categorias):
    """Cria um card para variáveis qualitativas"""
    return html.Div([
        html.H5(titulo, style={'margin': '0 0 8px 0', 'color': COLOR_TEXT_MAIN, 'fontSize': '15px'}),
        html.Span("Categórico (Object)", style={'backgroundColor': COLOR_BG, 'color': COLOR_PRIMARY, 'padding': '4px 8px', 'borderRadius': '4px', 'fontSize': '12px', 'fontWeight': 'bold'}),
        html.P(categorias, style={'marginTop': '12px', 'fontSize': '14px', 'color': COLOR_TEXT_MUTED})
    ], style={'backgroundColor': COLOR_BG, 'padding': '16px', 'borderRadius': '6px', 'border': f'1px solid {COLOR_BORDER}'})

# ── ESTRUTURA LATERAL (SIDEBAR DE NAVEGAÇÃO INTERNA) ──
sidebar_docs = html.Div([
    html.H4("Índice", style={'color': COLOR_PRIMARY, 'marginBottom': '20px', 'fontSize': '18px'}),
    html.Ul([
        html.Li("Resumo Geral", style={'marginBottom': '12px', 'color': COLOR_TEXT_MUTED, 'cursor': 'pointer', 'fontWeight': '500'}),
        html.Li("1. Perfil Demográfico", style={'marginBottom': '12px', 'color': COLOR_TEXT_MUTED, 'cursor': 'pointer'}),
        html.Li("2. Despesas Mensais", style={'marginBottom': '12px', 'color': COLOR_TEXT_MUTED, 'cursor': 'pointer'}),
        html.Li("3. Saúde Financeira", style={'marginBottom': '12px', 'color': COLOR_TEXT_MUTED, 'cursor': 'pointer'}),
        html.Li("4. Economia Potencial", style={'marginBottom': '12px', 'color': COLOR_TEXT_MUTED, 'cursor': 'pointer'}),
    ], style={'listStyleType': 'none', 'padding': '0', 'margin': '0', 'fontSize': '15px'})
], style={
    'width': '250px', 
    'backgroundColor': COLOR_BG, 
    'borderRight': f'1px solid {COLOR_BORDER}',
    'padding': '30px', 
    'flexShrink': '0' ,
    'position': 'sticky', # Faz a sidebar grudar na tela
    'top': '0',           # Gruda exatamente no topo do container
    'height': '100vh'
})

# ── CONTEÚDO PRINCIPAL DA DOCUMENTAÇÃO ──
conteudo_docs = html.Div([
    
    html.H1("Documentação do Dataset", style={'color': COLOR_TEXT_MAIN, 'marginTop': '0', 'marginBottom': '8px'}),
    html.P("Referência técnica estruturada para análise e modelagem de dados.", style={'color': COLOR_TEXT_MUTED, 'marginBottom': '30px'}),

    # Resumo Geral (Badges)
    html.Div([
        html.Div([html.Strong("Total de Registros: "), "20.000 linhas"], style={'padding': '12px 20px', 'backgroundColor': COLOR_BG, 'color': COLOR_PRIMARY, 'borderRadius': '6px', 'border': f'1px solid {COLOR_PRIMARY}'}),
        html.Div([html.Strong("Total de Colunas: "), "27"], style={'padding': '12px 20px', 'backgroundColor': COLOR_SURFACE, 'border': f'1px solid {COLOR_BORDER}', 'borderRadius': '6px'}),
        html.Div([html.Strong("Formatos: "), "Numérico & Categórico"], style={'padding': '12px 20px', 'backgroundColor': COLOR_SURFACE, 'border': f'1px solid {COLOR_BORDER}', 'borderRadius': '6px'}),
    ], style={'display': 'flex', 'gap': '15px', 'marginBottom': '40px', 'flexWrap': 'wrap'}),

    # SEÇÃO 1: PERFIL DEMOGRÁFICO
    html.H3("1. Perfil Demográfico e Socioeconômico", style={'borderBottom': f'2px solid {COLOR_PRIMARY}', 'paddingBottom': '10px', 'color': COLOR_TEXT_MAIN}),
    html.P("Descreve as características qualitativas e quantitativas dos indivíduos analisados.", style={'color': COLOR_TEXT_MUTED, 'marginBottom': '20px'}),
    
    html.H4("Dados Qualitativos (Categóricos)", style={'color': COLOR_TEXT_MAIN, 'marginTop': '20px'}),
    html.Div([
        card_categorico("Occupation (Ocupação)", "Aposentado (Retired), Autônomo (Self-Employed), Profissional (Professional) e Estudante (Student)."),
        card_categorico("City_Tier (Classificação)", "Tier 1 (Metrópoles), Tier 2 (Cidades Médias) e Tier 3 (Cidades Menores)."),
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(300px, 1fr))', 'gap': '20px', 'marginBottom': '30px'}),

    html.H4("Dados Quantitativos (Numéricos)", style={'color': COLOR_TEXT_MAIN}),
    html.Div([
        card_metrica("Income (Renda Mensal)", "41.585,50", "40.014,54", "1.301,18", "1.079.728,00"),
        card_metrica("Age (Idade)", "41,03 anos", "13,58", "18 anos", "64 anos"),
        card_metrica("Dependents (Dependentes)", "1,99", "1,42", "0", "4"),
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '20px', 'marginBottom': '40px'}),

    # SEÇÃO 2: DESPESAS MENSAIS
    html.H3("2. Detalhamento de Despesas Mensais", style={'borderBottom': f'2px solid {COLOR_PRIMARY}', 'paddingBottom': '10px', 'color': COLOR_TEXT_MAIN}),
    html.P("Informações numéricas (float64) sobre os custos de vida. O desvio padrão elevado em categorias como 'Rent' indica grande dispersão financeira na amostra.", style={'color': COLOR_TEXT_MUTED, 'marginBottom': '20px'}),
    html.Div([
        card_metrica("Rent (Aluguel)", "9.115,49", "9.254,23", "235,36", "215.945,67"),
        card_metrica("Loan Repayment (Empréstimos)", "2.049,80", "4.281,79"),
        card_metrica("Groceries (Mercado)", "5.205,66", "5.035,95"),
        card_metrica("Transport (Transporte)", "2.704,46", "2.666,35"),
        card_metrica("Eating Out (Restaurantes)", "1.461,85", "1.481,66"),
        card_metrica("Entertainment (Lazer)", "1.448,85", "1.489,02"),
        card_metrica("Insurance (Seguros)", "1.455,02", "1.492,94"),
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '20px', 'marginBottom': '40px'}),

    # SEÇÃO 3: SAÚDE FINANCEIRA
    html.H3("3. Metas e Indicadores de Saúde Financeira", style={'borderBottom': f'2px solid {COLOR_PRIMARY}', 'paddingBottom': '10px', 'color': COLOR_TEXT_MAIN}),
    html.Div([
        card_metrica("Desired Savings (Meta de Economia)", "4.982,87", "7.733,47"),
        card_metrica("Disposable Income (Renda Disponível)", "10.647,36", "11.740,64", "-5.400,78", "377.060,22"),
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(350px, 1fr))', 'gap': '20px', 'marginBottom': '40px'}),

    # SEÇÃO 4: ECONOMIA POTENCIAL
    html.H3("4. Economia Potencial (Potential Savings)", style={'borderBottom': f'2px solid {COLOR_PRIMARY}', 'paddingBottom': '10px', 'color': COLOR_TEXT_MAIN}),
    html.P("Valores estimados de redução de gastos possíveis por categoria.", style={'color': COLOR_TEXT_MUTED, 'marginBottom': '20px'}),
    html.Div([
        card_metrica("Groceries (Mercado)", "912,19", "1.038,88"),
        card_metrica("Transport (Transporte)", "473,04", "537,22"),
        card_metrica("Utilities (Utilidades)", "436,33", "503,20"),
        card_metrica("Eating Out (Restaurantes)", "254,96", "296,05"),
        card_metrica("Entertainment (Lazer)", "254,03", "299,97"),
        card_metrica("Miscellaneous (Diversos)", "144,90", "169,16"),
        card_metrica("Education (Educação)", "62,41", "98,84"),
        card_metrica("Healthcare (Saúde)", "41,52", "53,15"),
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '15px', 'marginBottom': '60px'}),

], style={
    'flex': '1', # Ocupa todo o espaço restante ao lado da sidebar
    'padding': '40px',
    
    'backgroundColor': COLOR_SURFACE
})

# ── EXPORTAÇÃO DO LAYOUT FINAL ──
# Este é o container que une a Sidebar Lateral ao Conteúdo Principal usando Flexbox
layout = html.Div([
    sidebar_docs,
    conteudo_docs
], style={'display': 'flex', 'height': '100vh', 'fontFamily': 'Segoe UI, Roboto, sans-serif'})