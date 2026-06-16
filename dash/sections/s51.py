import dash
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
COLOR_PLOT_FILL = "#7f7f7f"

CARD_STYLE = {
    'backgroundColor': BG_CARD,
    'borderRadius': '12px',
    'padding': '32px',
    'marginBottom': '40px',
    'border':'1px solid #E5E7EB',                 # Borda cinza bem clara e sutil
    'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.04)'
}

TEXT_P_STYLE = {
    'color': TEXT_MUTED, 
    'marginBottom': '24px', 
    'lineHeight': '1.7',
    'fontSize': '16px',
    'textAlign': 'justify'
}

def criar_bloco_estatistico_estatico(h_val, p_val, veredito, cor_veredito):
    """Função auxiliar para gerar as caixas de texto estático no layout."""
    return html.Div([
        html.H4("Teste de Hipótese (Kruskal-Wallis)", style={'color': COLOR_ACCENT, 'marginBottom': '16px'}),
        html.Div([
            html.Div([html.Span("Estatística H: "), html.Strong(h_val)], style={'marginBottom': '8px', 'color': TEXT_MAIN}),
            html.Div([html.Span("P-Valor: "), html.Strong(p_val)], style={'color': TEXT_MAIN})
        ], style={'backgroundColor': 'rgba(0,0,0,0.03)', 'padding': '16px', 'borderRadius': '8px', 'marginBottom': '16px'}),
        
        html.Div([html.Strong("Veredito: ", style={'color': TEXT_MAIN}), html.Span(veredito, style={'color': cor_veredito, 'fontWeight': 'bold'})])
    ], style={'marginTop': '24px', 'borderTop': f'1px solid {BORDER_COLOR}', 'paddingTop': '24px'})

def get_layout():
    """Retorna o layout estrutural da página. Os gráficos são injetados vazios."""
    return html.Div([
        
        html.H1("Objetivo da Análise", style={'color': TEXT_MAIN, 'fontSize': '36px', 'marginBottom': '16px'}),
        html.P("O objetivo desta análise é demonstrar que variáveis como o número de dependentes, profissão e idade não são fundamentais para definir se o usuário terá uma boa gestão do seu dinheiro.", 
               style={'color': TEXT_MUTED, 'fontSize': '18px', 'marginBottom': '40px', 'lineHeight': '1.6'}),
        
        html.Div([
            html.H3("Metodologia", style={'color': COLOR_ACCENT, 'fontSize': '20px', 'marginBottom': '12px'}),
            html.P("Analisaremos essas variáveis em relação ao Savings Ratio (Taxa de Economia), que busca identificar quanto da renda está sendo 'perdido' em gastos supérfluos.", style=TEXT_P_STYLE),
            
            html.H4("O que o seu 'Percentual' realmente significa?", style={'color': TEXT_MAIN, 'fontSize': '16px', 'marginBottom': '12px'}),
            html.Ul([
                html.Li([html.Strong("Percentual ALTO (Ex: 40%): ", style={'color': COLOR_ERROR}), "Indica que a pessoa é ineficiente financeiramente. Ela gasta quase metade do que ganha com itens não essenciais."]),
                html.Li([html.Strong("Percentual BAIXO (Ex: 5%): ", style={'color': COLOR_SUCCESS}), "Indica que a pessoa é eficiente. O orçamento está tão ajustado que quase não sobra margem para cortar gastos supérfluos."])
            ], style={'color': TEXT_MUTED, 'lineHeight': '1.8'}),
        ], style=CARD_STYLE),

        html.Div([
            html.H2("4.1. Análise: Potencial de Economia x Idade", style={'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'}),
            html.P("A análise a seguir tenta provar que a idade não influencia diretamente no Potencial de economia. Embora o grupo de idosos possua uma amostra menor, a densidade do comportamento financeiro (onde o violino é mais 'gordo') é muito parecida com as demais idades.", style=TEXT_P_STYLE),
            
            # Gráfico dinâmico
            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-idade-container')),
            
            # Texto Estatístico Estático
            criar_bloco_estatistico_estatico(
                h_val="6.5098", 
                p_val="0.0893", 
                veredito="As faixas etárias são estatisticamente iguais.", 
                cor_veredito=COLOR_SUCCESS
            )
        ], style=CARD_STYLE),

        # 4.2 OCUPAÇÃO
        html.Div([
            html.H2("4.2. Análise: Potencial de Economia x Ocupação", style={'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'}),
            html.P("Nesta análise, verificamos a relação entre a profissão e a gestão da renda. O padrão observado anteriormente repete-se: a maior densidade de dados converge para a mesma percentagem, independentemente do cargo ocupado.", style=TEXT_P_STYLE),
            
            # Gráfico dinâmico
            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-ocupacao-container')),
            
            # Texto Estatístico Estático
            criar_bloco_estatistico_estatico(
                h_val="6.0737", 
                p_val="0.1081", 
                veredito="As ocupações são estatisticamente iguais.", 
                cor_veredito=COLOR_SUCCESS
            )
        ], style=CARD_STYLE),

        # 4.3 DEPENDENTES
        html.Div([
            html.H2("4.3. Análise: Potencial de Economia x Dependentes", style={'color': TEXT_MAIN, 'fontSize': '28px', 'marginBottom': '16px'}),
            html.P("A análise indica que pessoas sem dependentes tendem a apresentar melhores índices de potencial de economia. No entanto, ao observar a densidade, notamos que a grande massa de dados se concentra de forma muito semelhante entre os grupos.", style=TEXT_P_STYLE),
            
            # Gráfico dinâmico
            dcc.Loading(color=COLOR_ACCENT, children=html.Div(id='grafico-dependentes-container')),
            
            # Texto Estatístico Estático
            criar_bloco_estatistico_estatico(
                h_val="82.1886", 
                p_val="0.0000", 
                veredito="Existe influência do número de dependentes.", 
                cor_veredito=COLOR_ERROR 
            )
        ], style=CARD_STYLE),

        #  CONCLUSÃO GERAL 
        html.Div([
            html.H2("Conclusão Geral da Análise Demográfica", style={'color': COLOR_ACCENT, 'fontSize': '28px', 'marginBottom': '20px'}),
            html.P("Após a realização dos testes estatísticos de Kruskal-Wallis, consolidamos as seguintes descobertas:", style=TEXT_P_STYLE),
            
            html.H4("Fatores Não-Determinantes (Ocupação e Idade)", style={'color': TEXT_MAIN, 'fontSize': '18px', 'marginBottom': '8px'}),
            html.P("As análises provaram que a Ocupação e a Faixa Etária não possuem influência estatisticamente significativa sobre a eficiência financeira. Isso quebra o paradigma de que certos cargos ou maior maturidade resultariam em uma gestão mais consciente.", style=TEXT_P_STYLE),
            
            html.H4("O Fator Determinante (Número de Dependentes)", style={'color': TEXT_MAIN, 'fontSize': '18px', 'marginBottom': '8px'}),
            html.P("A variável Número de Dependentes revelou-se um fator determinante no comportamento de desperdício. Mesmo sendo uma diferença sutil numericamente (aprox. 0.22 p.p.), ela é estatisticamente sólida devido ao robusto volume da amostra.", style=TEXT_P_STYLE),
            
            html.Div([
                html.Strong("Análise Social: ", style={'color': COLOR_ACCENT}),
                html.Span("Esta análise reforça a relevância de um planejamento familiar sólido. A convergência dos resultados nas outras variáveis indica uma carência generalizada de instrução sobre finanças que nivela o comportamento de gastos por toda a população, independentemente da fase da vida ou profissão.", style={'color': TEXT_MAIN, 'lineHeight': '1.6'})
            ], style={'backgroundColor': 'rgba(29, 18, 82, 0.05)', 'padding': '20px', 'borderRadius': '8px', 'borderLeft': f'4px solid {COLOR_ACCENT}'})
            
        ], style=CARD_STYLE)
        
    ], style={'maxWidth': '1100px', 'margin': '0 auto'})


def _gerar_apenas_grafico(df, var_x, var_y, labels_x, titulo, order_cat=None):
    """Gera apenas a figura Plotly, sem textos dinâmicos."""
    
    fig = px.violin(
        df.sample(min(300, len(df)), random_state=42).reset_index(drop=True), x=var_x, y=var_y, 
        color_discrete_sequence=[COLOR_PLOT_FILL], 
        box=True, points='all', 
        category_orders={var_x: order_cat} if order_cat else None,
        labels={var_x: labels_x, var_y: "% de Economia Supérflua"},
        title=titulo
    )

    fig.update_traces(
        scalemode='count', opacity=0.6, line_width=1.5, line_color='black', 
        meanline_visible=True, meanline_width=3, marker=dict(opacity=0.3, size=3)
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TEXT_MAIN, family="Segoe UI, sans-serif"),
        margin=dict(t=50, b=40, l=40, r=20), hovermode="x unified",
        title_font=dict(size=16, color=COLOR_ACCENT),
        xaxis=dict(showgrid=False, showline=True, linecolor=BORDER_COLOR),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=False)
    )

    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def register_callbacks(app):
    
    def pre_processar_dados(data):
        """Helpers para calcular coluna base se necessário"""
        if not data: return None
        df = pd.DataFrame(data)
        if 'Waste_Ratio' not in df.columns:
            if all(col in df.columns for col in ['Eating_Out', 'Entertainment', 'Miscellaneous', 'Income']):
                df['Waste_Ratio'] = ((df['Eating_Out'] + df['Entertainment'] + df['Miscellaneous']) / df['Income'].replace(0, 1)) * 100
            else:
                return None
        return df

    
    @app.callback(Output('grafico-idade-container', 'children'), Input('filtered-data-store', 'data'))
    def render_idade(data):
        df = pre_processar_dados(data)
        if df is None: return html.Div("Erro de dados.", style={'color': COLOR_ERROR})
        
        bins = [18, 31, 46, 61, 200]
        labels = ['Jovem Adulto', 'Adulto', 'Sênior', 'Idoso']
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        df_clean = df.dropna(subset=['Age_Group', 'Waste_Ratio'])
        
        return _gerar_apenas_grafico(
            df=df_clean, var_x='Age_Group', var_y='Waste_Ratio', 
            labels_x="Faixa Etária", titulo="Distribuição por Faixa Etária (18+ anos)", order_cat=labels
        )

    
    @app.callback(Output('grafico-ocupacao-container', 'children'), Input('filtered-data-store', 'data'))
    def render_ocupacao(data):
        df = pre_processar_dados(data)
        if df is None: return html.Div("Erro de dados.", style={'color': COLOR_ERROR})
        
        df_clean = df.dropna(subset=['Occupation', 'Waste_Ratio']).copy()
        df_clean['Occupation'] = df_clean['Occupation'].astype(str).str.replace('_', ' ')
        ordem = ['Student', 'Professional', 'Self Employed', 'Retired'] 
        
        return _gerar_apenas_grafico(
            df=df_clean, var_x='Occupation', var_y='Waste_Ratio', 
            labels_x="Tipo de Ocupação", titulo="Distribuição por Ocupação", order_cat=ordem
        )

    @app.callback(Output('grafico-dependentes-container', 'children'), Input('filtered-data-store', 'data'))
    def render_dependentes(data):
        df = pre_processar_dados(data)
        if df is None: return html.Div("Erro de dados.", style={'color': COLOR_ERROR})
        
        df_clean = df.dropna(subset=['Dependents', 'Waste_Ratio']).copy()
        df_clean['Dependents'] = pd.to_numeric(df_clean['Dependents'], errors='coerce').fillna(0).astype(int).astype(str)
        ordem = sorted(df_clean['Dependents'].unique(), key=int)
        
        return _gerar_apenas_grafico(
            df=df_clean, var_x='Dependents', var_y='Waste_Ratio', 
            labels_x="Número de Dependentes", titulo="Distribuição por Dependentes", order_cat=ordem
        )