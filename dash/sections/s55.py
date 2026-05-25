from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data import df_master # <--- Importação do dataset para ler as idades limites

# Identifica dinamicamente a menor e a maior idade presentes no df_master
min_age_dataset = int(df_master['Age'].min())
max_age_dataset = int(df_master['Age'].max())

layout = html.Div([
    html.H2("5.5 – Gasto com Dopamina e Imposto do Cansaço"),
    
    # Controles Dinâmicos Específicos da Seção
    html.Div([
        html.Div([
            html.Label("Visão da Análise:", style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='s55-view',
                options=[
                    {'label': ' 1. Evolução por Idade',          'value': 'idade'},
                    {'label': ' 2. Desvio por Ocupação',         'value': 'ocupacao'},
                    {'label': ' 3. Imposto do Cansaço por Tier', 'value': 'tier'},
                ],
                value='idade',
                inputStyle={'marginRight': '6px'},
                labelStyle={'display': 'inline-block', 'marginRight': '15px'},
            ),
        ], style={'marginBottom': '15px'}),

        html.Div([
            html.Label("Composição do Gasto com Dopamina:", style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='s55-categories',
                options=[
                    {'label': ' Comer Fora (Eating Out)', 'value': 'Eating_Out'},
                    {'label': ' Entretenimento (Entertainment)', 'value': 'Entertainment'},
                    {'label': ' Diversos (Miscellaneous)', 'value': 'Miscellaneous'}
                ],
                value=['Eating_Out', 'Entertainment', 'Miscellaneous'],
                inputStyle={'marginRight': '6px'},
                labelStyle={'display': 'inline-block', 'marginRight': '15px'},
            )
        ], style={'marginBottom': '15px'}),


        html.Div([
            html.Label("Idade de Maturidade Financeira (Ponto de Inflexão):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='s55-maturity-age',
                min=min_age_dataset,
                max=max_age_dataset,
                step=1,
                value=32,
                allow_direct_input= False,
                # Gera as marcações visuais de 5 em 5 anos baseando-se no teto do dataset
                marks={i: str(i) for i in range(min_age_dataset, max_age_dataset + 1, 5)},
            )
        ], style={'marginBottom': '15px'})
    ], style={'padding': '15px', 'backgroundColor': '#f0f4f8', 'borderRadius': '8px', 'marginBottom': '20px'}),
    
    dcc.Graph(id='s55-graph'),
])


def register_callbacks(app):
    @app.callback(
        Output('s55-graph', 'figure'),
        Input('filtered-data-store', 'data'),
        Input('s55-view', 'value'),
        Input('s55-categories', 'value'),
        Input('s55-maturity-age', 'value')
    )
    def update_s55(stored_data, view, categories, maturity_age):
        if not stored_data or not categories:
            return go.Figure().update_layout(title="Selecione ao menos uma categoria de gasto.")

        dff = pd.DataFrame(stored_data)
        
        for col in categories:
            if col not in dff.columns:
                dff[col] = 0
                
        dff['Dopamine_Spend'] = dff[categories].sum(axis=1)
        dff['Income_Safe'] = dff['Income'].replace(0, pd.NA) 
        dff['Fatigue_Tax'] = dff['Dopamine_Spend'] / dff['Income_Safe']

        if view == 'idade':
            df_age = dff.groupby('Age', as_index=False)['Dopamine_Spend'].mean()
            df_age = df_age.sort_values('Age')
            df_age['Tendencia'] = df_age['Dopamine_Spend'].rolling(window=5, min_periods=1, center=True).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_age['Age'], y=df_age['Dopamine_Spend'], 
                                     mode='lines+markers', name='Gasto Médio (Bruto)', 
                                     line=dict(color='lightgray', width=1), opacity=0.7))
            fig.add_trace(go.Scatter(x=df_age['Age'], y=df_age['Tendencia'], 
                                     mode='lines', name='Tendência Suavizada', 
                                     line=dict(color='#d62728', width=3)))
            
            fig.add_vline(x=maturity_age, line_dash="dash", line_color="gray")
            fig.add_annotation(x=maturity_age, y=df_age['Dopamine_Spend'].max(),
                               text="Maturidade Financeira", showarrow=False, xanchor="left", xshift=10)

            fig.update_layout(title="Análise de Gasto com Dopamina por Idade",
                              xaxis_title="Idade", yaxis_title="Gasto Total Médio ($)",
                              height=500, margin=dict(t=50, b=50, l=60, r=20),
                              hovermode='x unified')

        elif view == 'ocupacao':
            mean_geral = dff['Fatigue_Tax'].mean()
            df_occ = dff.groupby('Occupation', as_index=False)['Fatigue_Tax'].mean()
            df_occ['Desvio_Percentual'] = ((df_occ['Fatigue_Tax'] - mean_geral) / mean_geral) * 100
            df_occ = df_occ.sort_values('Desvio_Percentual')

            fig = px.bar(df_occ, x='Desvio_Percentual', y='Occupation', orientation='h',
                         title="Desvio do Imposto do Cansaço vs Média Geral por Ocupação",
                         labels={'Desvio_Percentual': 'Desvio Percentual (%)', 'Occupation': 'Ocupação'},
                         color='Desvio_Percentual', color_continuous_scale='RdBu')
            
            fig.update_layout(
                height=500,
                margin=dict(t=50, b=50, l=140, r=80),
                coloraxis_showscale=False,
                xaxis=dict(automargin=True),
                yaxis=dict(automargin=True),
            )
            fig.add_vline(x=0, line_width=2, line_color="black")
            fig.update_traces(texttemplate='%{x:+.2f}%', textposition='outside')

        else:
            dff['Fase_de_Vida'] = dff['Age'].apply(lambda x: f'Antes dos {maturity_age}' if x < maturity_age else f'{maturity_age} anos ou mais')
            df_tier = dff.groupby(['City_Tier', 'Fase_de_Vida'], as_index=False)['Fatigue_Tax'].mean()
            
            fig = px.line(df_tier, x='City_Tier', y='Fatigue_Tax', color='Fase_de_Vida', markers=True,
                          title=f"Imposto do Cansaço por City Tier (Corte aos {maturity_age} anos)",
                          labels={'City_Tier': 'City Tier', 'Fatigue_Tax': 'Imposto do Cansaço (% da Renda)'})
            
            fig.update_traces(line=dict(width=3), marker=dict(size=8))
            fig.update_layout(height=500, margin=dict(t=50, b=50, l=60, r=20), yaxis_tickformat='.1%')

        return fig