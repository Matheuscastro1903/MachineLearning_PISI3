import dash
from dash import dcc, html, Input, Output

from data import df_master 
from sections import s51, s52, s53, s54, s55, s56_m1

# Tratamento de segurança: remove underscores caso existam no banco para ficar bonito no layout
city_tiers = [str(c).replace('_', ' ') for c in df_master['City_Tier'].unique()]
occupations = [str(o).replace('_', ' ') for o in df_master['Occupation'].unique()]
min_income  = df_master['Income'].min()
max_income  = df_master['Income'].max()

app = dash.Dash(__name__, title="ThinkMoney Dashboard")

app.layout = html.Div([
    
    # Store global que vai guardar os dados filtrados para repassar às seções
    dcc.Store(id='filtered-data-store'),

    # Header - Identidade ThinkMoney
    html.Div([
        # html.Img(src='/assets/logo.png', style={'height': '30px'}), # Descomente quando tiver a logo na pasta assets
        html.H2("ThinkMoney"),
    ], className="header"),

    # Container Principal
    html.Div([
        
        # ── SIDEBAR DE NAVEGAÇÃO ──
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

        # ── ÁREA DE CONTEÚDO ──
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

            # Output de Debug discreto
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

    ], className="app-container"),
])


@app.callback(
    Output('filtered-data-store', 'data'),
    [Input('filter-city', 'value'),
     Input('filter-occupation', 'value'),
     Input('filter-income', 'value')]
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
        return "0 registros encontrados."
    return f"Base ativa: {len(stored_data)} registros."

@app.callback(
    [Output('section-s51', 'style'), Output('section-s52', 'style'),
     Output('section-s53', 'style'), Output('section-s54', 'style'),
     Output('section-s55', 'style'), Output('section-s56', 'style')],
    Input('sidebar-nav', 'value'),
)
def toggle_sections(selected):
    sections = ['s51', 's52', 's53', 's54', 's55', 's56']
    return tuple({'display': 'block'} if selected == s else {'display': 'none'} for s in sections)

s51.register_callbacks(app)
s52.register_callbacks(app)
s53.register_callbacks(app)
s54.register_callbacks(app)
s55.register_callbacks(app)
s56_m1.register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)