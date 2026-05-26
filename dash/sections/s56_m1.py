from dash import dcc, html, Input, Output, State, dash_table
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

def aplicar_modelo_em_df(df, model, scaler):
    df = df.copy()
    X = df[FEATURES]
    X_scaled = scaler.transform(X)
    df['probabilidade_risco'] = model.predict_proba(X_scaled)[:, 1]
    df['classificacao_binaria'] = model.predict(X_scaled)

    def classificar_risco(p):
        if p < 0.30:
            return 'Baixo'
        if p < 0.60:
            return 'Médio'
        return 'Alto'

    df['nivel_risco'] = df['probabilidade_risco'].apply(classificar_risco)
    return df

def resumo_cards(df):
    total = len(df)
    alto = df[df['nivel_risco'] == 'Alto']
    exposicao_alto = alto['Loan_Repayment'].sum() if len(alto) else 0
    cards = html.Div(
        [
            html.Div("Resumo da vulnerabilidade financeira", style={
                'gridColumn': '1 / -1',
                'fontWeight': 'bold',
                'fontSize': '16px',
                'marginBottom': '4px',
            }),
            html.Div([
                html.Div("Clientes", className="summary-grid-label"),
                html.Div(f"{total:,}", className="summary-grid-value"),
            ], className="summary-grid-cell"),
            html.Div([
                html.Div("Vulneráveis (1)", className="summary-grid-label"),
                html.Div(f"{(df['classificacao_binaria']==1).sum():,}", className="summary-grid-value"),
            ], className="summary-grid-cell"),
            html.Div([
                html.Div("Seguros (0)", className="summary-grid-label"),
                html.Div(f"{(df['classificacao_binaria']==0).sum():,}", className="summary-grid-value"),
            ], className="summary-grid-cell"),
            html.Div([
                html.Div("Exposição — Alto Risco", className="summary-grid-label"),
                html.Div(f"R$ {exposicao_alto:,.0f}", className="summary-grid-value"),
            ], className="summary-grid-cell"),
        ],
        style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(2, minmax(0, 1fr))',
            'gap': '12px',
            'marginBottom': '12px',
        }
    )
    return cards

layout = html.Div([
    html.H2("M1 — Previsão de Vulnerabilidade Financeira"),

    html.Div([
        html.Div([
            html.Label("Fonte de dados: dataset interno do dashboard"),
            html.Br(),
            html.Label("Threshold (probabilidade) — ajustar sensibilidade"),
            dcc.Slider(id="m1-threshold-slider", min=0.0, max=1.0, step=0.01, value=0.5,
                       marks={0.0: "0.0", 0.3: "0.3", 0.6: "0.6", 1.0: "1.0"}),
            html.Br(),
            html.Button("Executar modelo", id="m1-run-btn"),
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),

        html.Div([
            html.Div(id="m1-cards", style={'display': 'none'}),
            html.Div([
                html.Label("Visualizar:"),
                dcc.RadioItems(
                    id='m1-view',
                    options=[
                        {'label': ' Tabela de resultados', 'value': 'table'},
                        {'label': ' Distribuição por nível (Pizza)', 'value': 'pie'},
                        {'label': ' Histograma de probabilidades', 'value': 'hist'},
                        {'label': ' Mostrar todos', 'value': 'all'},
                    ],
                    value='table',
                    inline=True,
                    inputStyle={'marginRight': '6px'},
                    labelStyle={'marginRight': '12px'},
                ),
            ], style={'marginBottom': '8px'}),

            html.Div(dcc.Graph(id="m1-pie"), id='m1-pie-container', style={'display': 'none'}),
            html.Div(dcc.Graph(id="m1-hist"), id='m1-hist-container', style={'display': 'none'}),
        ], style={"width": "68%", "display": "inline-block", "padding": "10px"}),
    ]),

    html.Hr(),
    html.Div(id='m1-topn-container', style={'display': 'none'}, children=[
         html.Div([
            html.H4("Top N clientes por probabilidade"),
            html.Ul(id="m1-topn"),
        ], style={"width": "68%", "display": "inline-block", "padding": "10px"}),
    ]),

    html.Div(id='m1-table-container', style={'display': 'none'}, children=[
        html.H4("Tabela de resultados"),
        dash_table.DataTable(
            id="m1-table",
            page_size=10,
            sort_action="native",
            style_table={'overflowX': 'auto'},
            style_cell={
                'minWidth': '130px',
                'width': '130px',
                'maxWidth': '220px',
                'whiteSpace': 'nowrap',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
        ),
    ]),

    dcc.Store(id="m1-store-df"),
])

def register_callbacks(app):
    @app.callback(
        Output("m1-pie", "figure"),
        Output("m1-hist", "figure"),
        Output("m1-table", "data"),
        Output("m1-table", "columns"),
        Output("m1-topn", "children"),
        Output("m1-cards", "children"),
        Output("m1-store-df", "data"),
        Input("m1-run-btn", "n_clicks"),
        State("m1-threshold-slider", "value"),
        prevent_initial_call=True,
    )
    def run_model(n_clicks, threshold):
        # carregar dados do dataset interno do dashboard
        try:
            df = pd.read_csv(DEFAULT_DATA, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(DEFAULT_DATA, encoding="latin-1", on_bad_lines="skip")

        faltando = [c for c in FEATURES if c not in df.columns]
        if faltando:
            empty_fig = px.scatter(title=f"Colunas ausentes: {', '.join(faltando)}")
            return empty_fig, empty_fig, [], [], [], html.Div(f"Colunas faltando: {', '.join(faltando)}"), {}

        model, scaler = carregar_modelo()
        df_pred = aplicar_modelo_em_df(df, model, scaler)

        pie = px.pie(df_pred, names="nivel_risco", title="Distribuição por nível de risco")
        hist = px.histogram(df_pred, x="probabilidade_risco", nbins=30, title="Histograma de probabilidades")

        output_cols = ['probabilidade_risco', 'classificacao_binaria', 'nivel_risco'] + FEATURES
        table_data = df_pred[output_cols].to_dict("records")
        table_columns = [{"name": c, "id": c} for c in output_cols]

        topn = df_pred.sort_values("probabilidade_risco", ascending=False).head(10)
        topn_children = [html.Li(f"{i+1}. Prob {row['probabilidade_risco']:.3f} — Empréstimo R$ {row['Loan_Repayment']:,.0f}") for i, (_, row) in enumerate(topn.iterrows())]

        cards = resumo_cards(df_pred)

        stored = df_pred[output_cols].to_json(orient="records", force_ascii=False)

        return pie, hist, table_data, table_columns, topn_children, cards, stored


    @app.callback(
        Output('m1-pie-container', 'style'),
        Output('m1-hist-container', 'style'),
        Output('m1-cards', 'style'),
        Output('m1-topn-container', 'style'),
        Output('m1-table-container', 'style'),
        Input('m1-store-df', 'data'),
        Input('m1-view', 'value'),
    )
    def toggle_m1_view(stored_data, view):
        hide = {'display': 'none'}
        show = {}
        if not stored_data:
            return hide, hide, hide, hide, hide
        if view == 'all':
            return show, show, show, show, show
        if view == 'pie':
            return show, hide, show, show, hide
        if view == 'hist':
            return hide, show, show, show, hide
        if view == 'table':
            return hide, hide, show, show, show
        return show, show, show, show, show
