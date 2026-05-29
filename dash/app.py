import dash
from dash import dcc, html, Input, Output,ctx

from data import df_master 
from sections import s51, s52, s53, s54, s55, s56_m1


from pages.home import tela_home
from pages.documentacao import conteudo_docs




app = dash.Dash(__name__, title="ThinkMoney Dashboard", suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Store(id='filtered-data-store'),

    # Header - Identidade ThinkMoney (O SEU CÓDIGO INTACTO)
    html.Div([
        html.Div([
            html.Img(src='/assets/logo.png', style={'height': '60px', 'marginRight': '10px'}),
            html.H3("ThinkMoney", style={'margin': '0', 'color': "#FFFFFF"}),
        ], style={'display': 'flex', 'alignItems': 'center'}),

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
        'borderBottom': '1px solid #eaeaea'
    }),
    
    # ── A SOLUÇÃO PARTE 1 ESTÁ AQUI ──
    # O container de páginas NÃO pode ter padding nem margin!
    html.Div(id='page-content', style={'margin': '0', 'padding': '0', 'width': '100%'})
    
], style={'margin': '0', 'padding': '0'}) # O container PAI de todos também sem margens

# ── ROTEADOR CENTRAL ──
@app.callback(
    Output('page-content', 'children'), # Nós vamos alterar os 'filhos' dessa div
    [Input('btn-home', 'n_clicks'),
     Input('btn-docs', 'n_clicks'),
     Input('btn-eda', 'n_clicks'),
     Input('btn-ml', 'n_clicks')]
)
def mudar_pagina(b1, b2, b3, b4):
    # O 'ctx.triggered_id' é o identificador mágico do Dash. 
    # Ele descobre exatamente qual dos 4 botões disparou essa função.
    botao_clicado = ctx.triggered_id
    
    if botao_clicado == 'btn-eda':
        # Aqui você retornaria o layout importado da sua página de EDA.
        # Exemplo: return eda_page.layout
        return tela_home
        
    elif botao_clicado == 'btn-ml':
        # Exemplo: return s56_m1.layout
        return tela_home
        
    elif botao_clicado == 'btn-docs':
        return conteudo_docs
        
    else:
        # Cai aqui se o botão Home for clicado OU quando o aplicativo acabar de abrir
        return tela_home

if __name__ == '__main__':
    app.run(debug=True)