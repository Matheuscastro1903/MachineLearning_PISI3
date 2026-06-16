import dash
from dash import dcc, html, Input, Output,ctx

from data import df_master 
from sections import s51, s52, s53, s54, s55, s56_m1

from pages.home import tela_home
from pages.documentacao import conteudo_docs
from pages.analise import layout_eda,register_eda_callbacks
from pages.ml import layout as layout_ml, register_ml_callbacks

app = dash.Dash(__name__, title="ThinkMoney Dashboard", suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Store(id='filtered-data-store'),


    html.Div([
        html.Div([
            html.Img(src='/assets/logo.png', style={'height': '50px', 'marginRight': '12px'}),
            html.H3("ThinkMoney", style={
                'margin': '0', 
                'color': "#FFFFFF", 
                'fontWeight': '800',
                'letterSpacing': '1px',
                'fontFamily': 'Segoe UI, sans-serif'
            }),
        ], id='btn-logo', n_clicks=0, style={
            'display': 'flex', 
            'alignItems': 'center', 
            'cursor': 'pointer',
            'transition': 'opacity 0.2s ease',
        }),

        html.Div([
            html.Button('Home', id='btn-home', n_clicks=0, className='nav-btn'),
            html.Button('Documentação', id='btn-docs', n_clicks=0, className='nav-btn'),
            html.Button('EDA', id='btn-eda', n_clicks=0, className='nav-btn'),
            html.Button('Machine Learning', id='btn-ml', n_clicks=0, className='nav-btn'),
        ], style={'display': 'flex', 'gap': '12px'}) 

    ], className="header", style={
        'display': 'flex', 
        'justifyContent': 'space-between', 
        'alignItems': 'center', 
        'padding': '15px 30px',
        'backgroundColor': '#1D1252',
        'borderBottom': '1px solid rgba(255, 255, 255, 0.15)'
    }),
    
   
    html.Div(id='page-content', style={'margin': '0', 'padding': '0', 'width': '100%'})
    
], style={'margin': '0', 'padding': '0'}) 


@app.callback(
    [Output('page-content', 'children'),
     Output('btn-home', 'className'),
     Output('btn-docs', 'className'),
     Output('btn-eda', 'className'),
     Output('btn-ml', 'className')], 
    [Input('btn-logo', 'n_clicks'),
     Input('btn-home', 'n_clicks'),
     Input('btn-docs', 'n_clicks'),
     Input('btn-eda', 'n_clicks'),
    Input('btn-ml', 'n_clicks'),
    Input('btn-home-ml', 'n_clicks', allow_optional=True),
    Input('btn-home-eda', 'n_clicks', allow_optional=True)]
)
def mudar_pagina(b0, b1, b2, b3, b4, b5, b6):
    
    botao_clicado = ctx.triggered_id
    
    c_home = c_docs = c_eda = c_ml = 'nav-btn'
    
    if botao_clicado == 'btn-eda' or botao_clicado == 'btn-home-eda':
        c_eda = 'nav-btn-active'
        return layout_eda, c_home, c_docs, c_eda, c_ml
        
    elif botao_clicado == 'btn-ml' or botao_clicado == 'btn-home-ml':
        c_ml = 'nav-btn-active'
        return layout_ml, c_home, c_docs, c_eda, c_ml

    elif botao_clicado == 'btn-docs':
        c_docs = 'nav-btn-active'
        return conteudo_docs, c_home, c_docs, c_eda, c_ml
        
    else:
        c_home = 'nav-btn-active'
        return tela_home, c_home, c_docs, c_eda, c_ml

register_eda_callbacks(app)
register_ml_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)