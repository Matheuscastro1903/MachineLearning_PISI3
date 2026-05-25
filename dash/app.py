import dash
from dash import dcc, html, Input, Output
import pandas as pd

from data import df_master
from sections import s51, s52, s53, s54, s55

app = dash.Dash(__name__)

city_tiers = df_master['City_Tier'].unique().tolist()
occupations = df_master['Occupation'].unique().tolist()
min_income  = df_master['Income'].min()
max_income  = df_master['Income'].max()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
# Cada nova seção: adicionar uma entrada em 'options' e um html.Div abaixo.
sidebar = html.Div([
    html.H3("Seções", style={'marginBottom': '16px'}),
    dcc.RadioItems(
        id='sidebar-nav',
        options=[
            {'label': '5.1 – Waste Ratio',    'value': 's51'},
            {'label': '5.2 – Transporte',      'value': 's52'},
            {'label': '5.3 – Condicionantes',   'value': 's53'},
            {'label': '5.4 – Moradia',           'value': 's54'},
            {'label': '5.5 – Cansaço/Dopamina', 'value': 's55'},
        ],
        value='s51',
        labelStyle={'display': 'block', 'cursor': 'pointer'},
    ),
], style={
    'flex': '1 1 0%',
    'width': 'auto',
    'minWidth': '0',
    'minHeight': '80vh',
    'backgroundColor': '#f4f4f4',
    'padding': '16px',
    'borderRight': '1px solid #ddd',
    'flexShrink': '1',
})

# ── LAYOUT ─────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    html.H1("Dashboard Financeiro - Exploratória"),

    dcc.Store(id='filtered-data-store'),

    # Filtros globais
    html.Div([
        html.Label("Selecione o City Tier:"),
        dcc.Dropdown(
            id='filter-city',
            options=[{'label': city, 'value': city} for city in city_tiers],
            multi=True,
            placeholder="Todos os Tiers"
        ),
        html.Label("Selecione a Ocupação:"),
        dcc.Dropdown(
            id='filter-occupation',
            options=[{'label': occ, 'value': occ} for occ in occupations],
            multi=True,
            placeholder="Todas as Ocupações"
        ),
        html.Label("Faixa de Renda (Income):"),
        dcc.RangeSlider(
            id='filter-income',
            min=min_income,
            max=max_income,
            step=1000,
            marks={int(min_income): str(int(min_income)), int(max_income): str(int(max_income))},
            value=[min_income, max_income]
        ),
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'marginBottom': '10px'}),

    html.Div(id='debug-output', style={'padding': '0px 20px', 'borderBottom': '1px solid #ddd', 'marginBottom': '0'}),

    # Sidebar + conteúdo lado a lado
    html.Div([
        sidebar,
        html.Div(
            dcc.Loading(
                type='dot',
                color='#1f77b4',
                fullscreen=False,
                children=html.Div([
                    # ── cada seção fica em seu próprio Div com id único ──
                    # Para adicionar uma nova seção: html.Div(sXX.layout, id='section-sXX')
                    html.Div(s51.layout, id='section-s51'),
                    html.Div(s52.layout, id='section-s52'),
                    html.Div(s53.layout, id='section-s53'),
                    html.Div(s54.layout, id='section-s54'),
                    html.Div(s55.layout, id='section-s55'),
                ], style={
                    'width': '100%',
                    'minWidth': '0',
                    'padding': '24px',
                    'boxSizing': 'border-box',
                    'overflowY': 'auto',
                    'overflowX': 'hidden',
                }),
            ),
            style={'flex': '6 1 0%', 'minWidth': '0'},
        ),

    ], style={'display': 'flex', 'width': '100%'}),
])


# ── CALLBACKS ──────────────────────────────────────────────────────────────────
@app.callback(
    Output('filtered-data-store', 'data'),
    Input('filter-city',       'value'),
    Input('filter-occupation', 'value'),
    Input('filter-income',     'value'),
)
def update_store(selected_cities, selected_occupations, income_range):
    dff = df_master.copy()

    if selected_cities:
        dff = dff[dff['City_Tier'].isin(selected_cities)]

    if selected_occupations:
        dff = dff[dff['Occupation'].isin(selected_occupations)]

    if income_range:
        dff = dff[(dff['Income'] >= income_range[0]) & (dff['Income'] <= income_range[1])]

    return dff.to_dict('records')


@app.callback(
    Output('debug-output', 'children'),
    Input('filtered-data-store', 'data'),
)
def display_debug_info(stored_data):
    if not stored_data:
        return "Nenhum dado disponível."
    return f"Dados filtrados: {len(stored_data)} registros."


# Mostra a seção selecionada, esconde as demais.
# Para adicionar uma nova seção: Output('section-sXX', 'style') + lógica abaixo.
@app.callback(
    Output('section-s51', 'style'),
    Output('section-s52', 'style'),
    Output('section-s53', 'style'),
    Output('section-s54', 'style'),
    Output('section-s55', 'style'),
    Input('sidebar-nav', 'value'),
)
def toggle_sections(selected):
    sections = ['s51', 's52', 's53', 's54', 's55']
    return tuple({} if selected == s else {'display': 'none'} for s in sections)


s51.register_callbacks(app)
s52.register_callbacks(app)
s53.register_callbacks(app)
s54.register_callbacks(app)
s55.register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)
