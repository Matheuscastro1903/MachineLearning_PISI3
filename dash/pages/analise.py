import dash
from dash import dcc, html, Input, Output

from data import df_master 
from sections import s51, s52, s53, s54, s55

city_tiers = [str(c).replace('_', ' ') for c in df_master['City_Tier'].unique()]
occupations = [str(o).replace('_', ' ') for o in df_master['Occupation'].unique()]
min_income  = df_master['Income'].min()
max_income  = df_master['Income'].max()


BG_MAIN = "#FFFFFF"
BG_CARD = "#FFFFFF" 
BG_SIDEBAR = "#FFFFFF"
TEXT_MAIN = "#000000"
TEXT_MUTED = "#000000"
BORDER_COLOR = "rgba(0, 0, 0, 0.1)"


layout_eda = html.Div([
    
    
    dcc.Store(id='filtered-data-store'),

    
    html.Div([
        
       html.Div([
            html.H3("Exploração", style={
                'marginTop': '0', 'fontSize': '20px', 'color': TEXT_MAIN, 
                'marginBottom': '24px', 'fontWeight': 'bold'
            }),
            
            dcc.RadioItems(
                id='sidebar-nav',
                options=[
                    {'label': ' 5.1 – Waste Ratio',          'value': 's51'},
                    {'label': ' 5.2 – Transporte',           'value': 's52'},
                    {'label': ' 5.3 – Condicionantes',       'value': 's53'},
                    {'label': ' 5.4 – Moradia',              'value': 's54'},
                    {'label': ' 5.5 – Cansaço/Dopamina',     'value': 's55'},
                ],
                value='s51',
                className='custom-sidebar-menu', 
                labelStyle={
                    'display': 'flex', 'alignItems': 'center', 'cursor': 'pointer',
                    'padding': '12px 16px', 'marginBottom': '8px', 
                    'borderRadius': '6px', 'fontSize': '15px', 
                    'color': TEXT_MUTED, 'transition': 'all 0.2s ease'
                },
                inputStyle={'marginRight': '12px', 'cursor': 'pointer'} 
            ),
        ], style={
            'width': '280px', 
            'backgroundColor': '#FFFFFF', 
            'borderRadius': '16px',       # Bordas arredondadas
            'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.05)', 
            'padding': '32px 24px', 
            'flexShrink': '0', 
            'marginRight': '32px',        
            'height': 'fit-content'      
        }),
        
      
        html.Div([
            
        
            html.Div([
                html.H3("Filtros Globais", style={
                    'marginTop': '0', 'fontSize': '18px', 'marginBottom': '20px', 
                    'color': '#111827', 'fontWeight': 'bold'
                }),
                
                # Grid para os filtros ficarem lado a lado
                html.Div([
                    html.Div([
                        html.Label("City Tier:", style={'color': '#4B5563', 'fontSize': '14px', 'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='filter-city',
                            options=[{'label': city, 'value': city.replace(' ', '_')} for city in city_tiers],
                            multi=True,
                            placeholder="Todos os Tiers",
                            style={'backgroundColor': '#FFFFFF', 'color': '#333'} 
                        ),
                    ], style={'flex': '1', 'minWidth': '200px'}),
                    
                    html.Div([
                        html.Label("Ocupação:", style={'color': '#4B5563', 'fontSize': '14px', 'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='filter-occupation',
                            options=[{'label': occ, 'value': occ.replace(' ', '_')} for occ in occupations],
                            multi=True,
                            placeholder="Todas as Ocupações",
                            style={'backgroundColor': '#FFFFFF', 'color': '#333'}
                        ),
                    ], style={'flex': '1', 'minWidth': '200px'}),
                    
                    html.Div([
                        html.Label("Faixa de Renda:", style={'color': '#4B5563', 'fontSize': '14px', 'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.RangeSlider(
                            id='filter-income',
                            min=min_income,
                            max=max_income,
                            step=1000,
                            marks=None, 
                            tooltip={"placement": "bottom", "always_visible": True}, 
                            value=[min_income, max_income]
                        ),
                    ], style={'flex': '2', 'minWidth': '300px', 'paddingTop': '10px'}),
                ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap'})
                
            ], style={
                'backgroundColor': BG_CARD, 
                'padding': '24px 32px', 
                'borderRadius': '16px', 
                'border': f'1px solid {BORDER_COLOR}',                 
                'boxShadow': '0 2px 4px rgba(0,0,0,0.05)', 
                'marginBottom': '24px'
            }),

            
            html.Div(id='debug-output', style={'fontSize': '13px', 'color': TEXT_MUTED, 'textAlign': 'right', 'marginBottom': '20px'}),

            # Container das Seções
            html.Div([
                html.Div(s51.get_layout(), id='section-s51'),
                html.Div(s52.get_layout(), id='section-s52', style={'display': 'none'}),
                html.Div(s53.get_layout(), id='section-s53', style={'display': 'none'}),
                html.Div(s54.get_layout(), id='section-s54', style={'display': 'none'}),
                html.Div(s55.layout, id='section-s55', style={'display': 'none'}),
            ], style={
                'backgroundColor': BG_CARD, 'padding': '32px', 'borderRadius': '12px', 
                'border': f'1px solid {BORDER_COLOR}', 'minHeight': '600px'
            }),

        ], style={'flex': '1', 'padding': '40px', 'maxWidth': '1400px'})

    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '100%'})
], style={'backgroundColor': 'transparent', 'minHeight': '100vh', 'fontFamily': 'Segoe UI, sans-serif'})



def register_eda_callbacks(app):
    
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
         Output('section-s55', 'style')],
        Input('sidebar-nav', 'value'),
    )
    def toggle_sections(selected):
        sections = ['s51', 's52', 's53', 's54', 's55']
        return tuple({'display': 'block'} if selected == s else {'display': 'none'} for s in sections)

   
    s51.register_callbacks(app)
    s52.register_callbacks(app)
    s53.register_callbacks(app)
    s54.register_callbacks(app)
    s55.register_callbacks(app)