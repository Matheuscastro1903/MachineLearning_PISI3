# layout.py
from dash import html, dcc
from data import city_tiers, occupations, min_income, max_income

from sections import s51, s52, s53, s54, s55, s56_m1

container_principal = html.Div([
    
    #Container das análises
    html.Div([
        
        
        html.Div([
            html.H3("Exploração", style={'marginTop': '0', 'fontSize': '18px'}),
            dcc.RadioItems(
                id='sidebar-nav',
                options=[
                    {'label': ' 5.1 – Waste Ratio',          'value': 's51'},
                    {'label': ' 5.2 – Transporte',           'value': 's52'},
                    {'label': ' 5.3 – Condicionantes',       'value': 's53'},
                    {'label': ' 5.4 – Moradia',              'value': 's54'},
                    {'label': ' 5.5 – Cansaço/Dopamina',     'value': 's55'},
                    {'label': ' 5.6 – M1 (Vulnerabilidade)', 'value': 's56'},
                ],
                value='s51',
                labelStyle={'display': 'block', 'cursor': 'pointer', 'marginBottom': '12px', 'fontSize': '15px', 'color': '#333'},
            ),
        ], className="sidebar"),

       
        html.Div([
            
            # Card de Filtros Globais
            html.Div([
                html.H3("Filtros Globais", style={'marginTop': '0', 'fontSize': '18px', 'marginBottom': '16px'}),
                html.Div([
                    html.Div([
                        html.Label("City Tier:", className="filter-label"),
                        dcc.Dropdown(
                            id='filter-city',
                            options=[{'label': city, 'value': city.replace(' ', '_')} for city in city_tiers], # Repassa o valor com underscore para o Pandas filtrar certo
                            multi=True,
                            placeholder="Todos os Tiers"
                        ),
                    ]),
                    html.Div([
                        html.Label("Ocupação:", className="filter-label"),
                        dcc.Dropdown(
                            id='filter-occupation',
                            options=[{'label': occ, 'value': occ.replace(' ', '_')} for occ in occupations],
                            multi=True,
                            placeholder="Todas as Ocupações"
                        ),
                    ]),
                    html.Div([
                        html.Label("Faixa de Renda:", className="filter-label"),
                        dcc.RangeSlider(
                            id='filter-income',
                            min=min_income,
                            max=max_income,
                            step=1000,
                            marks={int(min_income): f"${int(min_income/1000)}k", int(max_income): f"${int(max_income/1000)}k"},
                            value=[min_income, max_income]
                        ),
                    ]),
                ], className="filters-grid")
            ], className="card"),

         
            html.Div(id='debug-output', style={'fontSize': '13px', 'color': '#888', 'textAlign': 'right', 'marginTop': '-10px'}),

            # Container das Seções
            html.Div([
                html.Div(s51.layout, id='section-s51'),
                html.Div(s52.layout, id='section-s52', style={'display': 'none'}),
                html.Div(s53.layout, id='section-s53', style={'display': 'none'}),
                html.Div(s54.layout, id='section-s54', style={'display': 'none'}),
                html.Div(s55.layout, id='section-s55', style={'display': 'none'}),
                html.Div(s56_m1.layout, id='section-s56', style={'display': 'none'}),
            ], className="card", style={'minHeight': '500px'}),

        ], className="content-area"),

    ], className="app-container"),])