from dash import html


ROXO_FUNDO = "#1D1252"
BRANCO = "#FFFFFF"

tela_home = html.Div(
    style={
        'backgroundColor': ROXO_FUNDO,
        'color': BRANCO,
        'width': '100%',       
        'minHeight': '100vh',  
        'padding': '120px 80px 80px 80px', 
        'position': 'relative',
        'overflow': 'hidden',
        'fontFamily': '"Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        'margin': '0',
        'boxSizing': 'border-box' 
    },
    children=[
        html.Div(style={
            'position': 'absolute', 'top': '0', 'right': '0', 'width': '60%', 'height': '100%', 
            'opacity': '0.1', 'pointerEvents': 'none',
            'background': 'radial-gradient(circle, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0) 70%)',
            'transform': 'translate(30%, -30%)'
        }),
        
        html.Div(
            style={
                'display': 'flex', 
                'flexDirection': 'row', 
                'alignItems': 'center', 
                'justifyContent': 'space-between',
                'gap': '80px',          
                'maxWidth': '1400px',   
                'margin': '0 auto'
            }, 
            children=[
            
            html.Div(style={'flex': '1.5'}, children=[
                
                html.Div(
                    style={
                        'display': 'inline-flex', 'alignItems': 'center', 'gap': '12px',
                        'padding': '8px 20px', 'borderRadius': '50px', 'marginBottom': '32px',
                        'backgroundColor': 'rgba(255, 255, 255, 0.05)',
                        'backdropFilter': 'blur(12px)',
                        'border': '1px solid rgba(255, 255, 255, 0.1)'
                    },
                    children=[
                        html.Span(style={'width': '10px', 'height': '10px', 'borderRadius': '50%', 'backgroundColor': '#4ADE80', 'boxShadow': '0 0 8px #4ADE80'}), # Ponto verde brilhante
                        html.Span("Algoritmo ao vivo", style={'fontSize': '14px', 'fontWeight': 'bold', 'letterSpacing': '2px'}) # Fonte maior
                    ]
                ),
                
                # Título Principal
                html.H1(
                    style={'fontSize': '72px', 'fontWeight': 'bold', 'lineHeight': '1.1', 'marginBottom': '32px', 'marginTop': '0'}, # Fonte de 56px para 72px
                    children=[
                        "Domine suas finanças com ",
                        html.Span("inteligência algorítmica", style={'textDecoration': 'underline', 'textDecorationColor': 'rgba(255,255,255,0.3)', 'textUnderlineOffset': '8px'})
                    ]
                ),
                
                # Parágrafo
                html.P(
                    "A plataforma definitiva para análise de dados financeiros em tempo real, impulsionada por algoritmos avançados e transparência total.", 
                    style={'fontSize': '22px', 'color': 'rgba(255, 255, 255, 0.8)', 'marginBottom': '48px', 'maxWidth': '700px', 'lineHeight': '1.6'} # Fonte de 18px para 22px
                ),
                
                # Botões
                html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
                    html.Button("Rodar Modelo", id='btn-home-ml', n_clicks=0, style={
                        'backgroundColor': BRANCO, 'color': ROXO_FUNDO, 'padding': '20px 40px', # Botões mais robustos
                        'fontWeight': 'bold', 'fontSize': '20px', 'borderRadius': '12px',       # Fonte de 16px para 20px
                        'border': 'none', 'cursor': 'pointer', 'boxShadow': '0 10px 15px -3px rgba(0, 0, 0, 0.2)'
                    }),
                    html.Button("Ver Análises", id='btn-home-eda', n_clicks=0, style={
                        'backgroundColor': 'rgba(255,255,255,0.05)', 'color': BRANCO, 'padding': '20px 40px', 
                        'fontWeight': 'bold', 'fontSize': '20px', 'borderRadius': '12px', 
                        'border': '1px solid rgba(255, 255, 255, 0.2)', 'cursor': 'pointer'
                    })
                ])
            ]),
            
            html.Div(style={'flex': '1', 'position': 'relative'}, children=[
                
                # O Card de Vidro
                html.Div(
                    style={
                        'backgroundColor': 'rgba(255, 255, 255, 0.1)',
                        'backdropFilter': 'blur(20px)',
                        'border': '1px solid rgba(255, 255, 255, 0.2)',
                        'borderRadius': '16px',
                        'padding': '32px', # Padding interno aumentado
                        'position': 'relative',
                        'zIndex': '10',
                        'boxShadow': '0 0 30px rgba(255, 255, 255, 0.1)'
                    }, 
                    children=[
                    
                    # Cabeçalho do Card
                    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '32px'}, children=[
                        html.Span("Análise de dados", style={'fontSize': '16px', 'fontWeight': 'bold', 'color': 'rgba(255,255,255,0.6)'}), # Fonte aumentada
                        
                    ]),
                    
                    html.Div(
                        style={'height': '260px', 'width': '100%', 'display': 'flex', 'alignItems': 'flex-end', 'gap': '8px', 'marginBottom': '32px'}, # Altura de 180px para 260px
                        children=[
                            html.Div(style={'flex': '1', 'backgroundColor': 'rgba(255,255,255,0.2)', 'height': '50%', 'borderRadius': '6px 6px 0 0'}),
                            html.Div(style={'flex': '1', 'backgroundColor': 'rgba(255,255,255,0.2)', 'height': '66%', 'borderRadius': '6px 6px 0 0'}),
                            html.Div(style={'flex': '1', 'backgroundColor': 'rgba(255,255,255,0.2)', 'height': '33%', 'borderRadius': '6px 6px 0 0'}),
                            html.Div(style={'flex': '1', 'backgroundColor': 'rgba(255,255,255,0.2)', 'height': '75%', 'borderRadius': '6px 6px 0 0'}),
                            html.Div(style={'flex': '1', 'backgroundColor': 'rgba(255,255,255,0.2)', 'height': '50%', 'borderRadius': '6px 6px 0 0'}),
                            html.Div(style={'flex': '1', 'backgroundColor': 'rgba(255,255,255,0.9)', 'height': '100%', 'borderRadius': '6px 6px 0 0', 'boxShadow': '0 0 20px rgba(255,255,255,0.5)'}),
                            html.Div(style={'flex': '1', 'backgroundColor': 'rgba(255,255,255,0.2)', 'height': '66%', 'borderRadius': '6px 6px 0 0'}),
                        ]
                    ),
                    
                    # Info do Rodapé do Card
                    html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'borderTop': '1px solid rgba(255,255,255,0.1)', 'paddingTop': '24px'}, children=[
                        html.Div([
                            html.P("Eficiência", style={'margin': '0', 'fontSize': '16px', 'color': 'rgba(255,255,255,0.6)', 'fontWeight': 'bold'}),
                            html.P("98.4%", style={'margin': '0', 'fontSize': '36px', 'fontWeight': 'bold', 'color': BRANCO}) # Fonte de 24px para 36px
                        ]),
                        html.Div(style={'textAlign': 'right'}, children=[
                            html.P("Carregamento de dados", style={'margin': '0', 'fontSize': '16px', 'color': 'rgba(255,255,255,0.6)', 'fontWeight': 'bold'}),
                            html.P("2.4 TB/s", style={'margin': '0', 'fontSize': '36px', 'fontWeight': 'bold', 'color': BRANCO}) # Fonte de 24px para 36px
                        ])
                    ])
                ]),
                
                # Decoração em baixo do card
                html.Div(style={
                    'position': 'absolute', 'bottom': '-50px', 'left': '-50px', 'width': '200px', 'height': '200px',
                    'backgroundColor': 'rgba(255,255,255,0.1)', 'filter': 'blur(80px)', 'borderRadius': '50%', 'zIndex': '1'
                })
            ])
        ])
    ]
)