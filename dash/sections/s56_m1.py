from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import joblib
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(ROOT_DIR, "..", "modelos", "modelo_joao_vulnerabilidade")
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_vulnerabilidade.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
DEFAULT_DATA = os.path.join(ROOT_DIR, "dataset", "data.csv")

FEATURES = [
    'Income', 'Age', 'Dependents', 'Loan_Repayment',
    'Eating_Out', 'Entertainment', 'Healthcare',
    'Rent', 'Groceries', 'Disposable_Income', 'Desired_Savings'
]

def carregar_modelo():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def aplicar_modelo_em_df(df, model, scaler, threshold):
    df = df.copy()
    X = df[FEATURES]
    X_scaled = scaler.transform(X)
    
    # Probabilidade exata gerada pelo Scikit-Learn
    df['probabilidade_risco'] = model.predict_proba(X_scaled)[:, 1]
    
    # Classificação dinâmica baseada no Slider (Threshold)
    df['classificacao_binaria'] = (df['probabilidade_risco'] >= threshold).astype(int)

    def classificar_risco(p):
        if p < 0.30: return 'Baixo'
        if p < 0.60: return 'Médio'
        return 'Alto'

    df['nivel_risco'] = df['probabilidade_risco'].apply(classificar_risco)
    return df

def resumo_cards(df):
    total = len(df)
    alto = df[df['nivel_risco'] == 'Alto']
    exposicao_alto = alto['Loan_Repayment'].sum() if len(alto) else 0
    
    card_style = {
        'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.05)', 'borderLeft': '4px solid #1D1252',
        'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center'
    }
    
    return html.Div([
        html.Div([
            html.Div("Clientes Analisados", style={'color': '#666', 'fontSize': '13px', 'fontWeight': '600', 'textTransform': 'uppercase'}),
            html.Div(f"{total:,}", style={'color': '#1D1252', 'fontSize': '24px', 'fontWeight': 'bold', 'marginTop': '4px'}),
        ], style=card_style),
        
        html.Div([
            html.Div("Seguros (Risco Baixo)", style={'color': '#666', 'fontSize': '13px', 'fontWeight': '600', 'textTransform': 'uppercase'}),
            html.Div(f"{(df['classificacao_binaria']==0).sum():,}", style={'color': '#2e7d32', 'fontSize': '24px', 'fontWeight': 'bold', 'marginTop': '4px'}),
        ], style={**card_style, 'borderLeftColor': '#2e7d32'}),
        
        html.Div([
            html.Div("Vulneráveis (Risco Alto)", style={'color': '#666', 'fontSize': '13px', 'fontWeight': '600', 'textTransform': 'uppercase'}),
            html.Div(f"{(df['classificacao_binaria']==1).sum():,}", style={'color': '#E54B4B', 'fontSize': '24px', 'fontWeight': 'bold', 'marginTop': '4px'}),
        ], style={**card_style, 'borderLeftColor': '#E54B4B'}),
        
        html.Div([
            html.Div("Exposição de Crédito (Alto Risco)", style={'color': '#666', 'fontSize': '13px', 'fontWeight': '600', 'textTransform': 'uppercase'}),
            html.Div(f"${exposicao_alto:,.0f}", style={'color': '#E54B4B', 'fontSize': '24px', 'fontWeight': 'bold', 'marginTop': '4px'}),
        ], style={**card_style, 'borderLeftColor': '#E54B4B'}),
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '16px', 'marginBottom': '24px'})


layout = html.Div([
    html.H2("5.6 – Previsão de Vulnerabilidade Financeira (M1)", 
            style={'marginBottom': '20px', 'color': '#1D1252', 'fontSize': '22px'}),

    # Painel de Controle e Execução
    html.Div([
        html.Div([
            html.Label("Sensibilidade do Modelo (Threshold):", style={'fontWeight': '600', 'display': 'block', 'marginBottom': '12px'}),
            dcc.Slider(id="m1-threshold-slider", min=0.0, max=1.0, step=0.01, value=0.5,
                       marks={0.0: "Mais Seguros", 0.5: "Padrão (0.5)", 1.0: "Mais Críticos"}),
        ], style={'flex': '1'}),
        
        html.Button("Executar Predição", id="m1-run-btn", 
                    style={'backgroundColor': '#1D1252', 'color': 'white', 'border': 'none', 
                           'padding': '12px 24px', 'borderRadius': '6px', 'fontWeight': 'bold', 
                           'cursor': 'pointer', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '30px', 'backgroundColor': '#f8f9fa', 'padding': '24px', 'borderRadius': '8px', 'marginBottom': '24px', 'border': '1px solid #eaeaea'}),

    # Área de Output (Cards Dinâmicos)
    html.Div(id="m1-cards", style={'display': 'none'}),

    # Opções de Visualização
    html.Div([
        html.Label("Perspectiva de Saída:", style={'fontWeight': '600', 'marginRight': '16px'}),
        dcc.RadioItems(
            id='m1-view',
            options=[
                {'label': ' Panorama Completo', 'value': 'all'},
                {'label': ' Níveis de Risco (Pizza)', 'value': 'pie'},
                {'label': ' Probabilidades (Histograma)', 'value': 'hist'},
            ],
            value='all',
            inline=True,
            inputStyle={'marginRight': '6px'},
            labelStyle={'marginRight': '20px', 'color': '#333', 'cursor': 'pointer'},
        ),
    ], style={'marginBottom': '20px', 'padding': '16px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'}),

    # Gráficos (Lado a Lado via Flexbox)
    html.Div([
        html.Div(dcc.Graph(id="m1-pie", config={'displayModeBar': False}), id='m1-pie-container', style={'flex': '1', 'display': 'none'}),
        html.Div(dcc.Graph(id="m1-hist", config={'displayModeBar': False}), id='m1-hist-container', style={'flex': '1', 'display': 'none'}),
    ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'marginBottom': '24px'}),

    dcc.Store(id="m1-store-df"),
])

def register_callbacks(app):
    @app.callback(
        Output("m1-pie", "figure"),
        Output("m1-hist", "figure"),
        Output("m1-cards", "children"),
        Output("m1-store-df", "data"),
        Input("m1-run-btn", "n_clicks"),
        State("m1-threshold-slider", "value"),
        prevent_initial_call=True,
    )
    def run_model(n_clicks, threshold):
        try:
            df = pd.read_csv(DEFAULT_DATA, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(DEFAULT_DATA, encoding="latin-1", on_bad_lines="skip")

        faltando = [c for c in FEATURES if c not in df.columns]
        if faltando:
            empty_fig = px.scatter(title=f"Erro: Colunas ausentes no dataset - {', '.join(faltando)}")
            return empty_fig, empty_fig, html.Div(f"Faltam as colunas: {', '.join(faltando)}", style={'color': 'red'}), {}

        model, scaler = carregar_modelo()
        df_pred = aplicar_modelo_em_df(df, model, scaler, threshold)

        color_map_risco = {'Baixo': '#1D1252', 'Médio': '#7B6FCD', 'Alto': '#E54B4B'}
        
        pie = px.pie(df_pred, names="nivel_risco", title="Distribuição por Nível de Risco",
                     color="nivel_risco", color_discrete_map=color_map_risco, hole=0.4)
        pie.update_traces(textposition='inside', textinfo='percent+label')
        pie.update_layout(plot_bgcolor='white', paper_bgcolor='white', showlegend=False)

        hist = px.histogram(df_pred, x="probabilidade_risco", nbins=30, 
                            title="Histograma de Probabilidades",
                            labels={'probabilidade_risco': 'Probabilidade de Risco (0 a 1)'},
                            color_discrete_sequence=['#594CA3'])
        hist.add_vline(x=threshold, line_dash="dash", line_color="#E54B4B", line_width=2,
                       annotation_text="Limiar (Threshold)", annotation_position="top right")
        hist.update_layout(plot_bgcolor='white', paper_bgcolor='white', yaxis_title="Nº de Clientes")
        hist.update_xaxes(showline=True, linewidth=1, linecolor='#e0e0e0', showgrid=False)
        hist.update_yaxes(showline=False, showgrid=True, gridwidth=1, gridcolor='#f0f0f0')

        cards = resumo_cards(df_pred)
        output_cols = ['probabilidade_risco', 'classificacao_binaria', 'nivel_risco'] + FEATURES
        df_pred['probabilidade_risco'] = df_pred['probabilidade_risco'].round(4)
        stored = df_pred[output_cols].to_json(orient="records", force_ascii=False)

        return pie, hist, cards, stored

    @app.callback(
        Output('m1-pie-container', 'style'),
        Output('m1-hist-container', 'style'),
        Output('m1-cards', 'style'),
        Input('m1-store-df', 'data'),
        Input('m1-view', 'value'),
    )
    def toggle_m1_view(stored_data, view):
        hide = {'display': 'none'}
        show_flex = {'flex': '1', 'display': 'block'}
        show_block = {'display': 'block'}
        
        if not stored_data:
            return hide, hide, hide
        if view == 'all':
            return show_flex, show_flex, show_block
        if view == 'pie':
            return show_flex, hide, show_block
        if view == 'hist':
            return hide, show_flex, show_block
        return show_flex, show_flex, show_block