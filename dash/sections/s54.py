from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

TIERS = ['Tier_1', 'Tier_2', 'Tier_3']
LABELS = {'City_Tier': 'City Tier', 'Rent': 'Aluguel ($)'}

layout = html.Div([
    html.H2("5.4 – Gasto com Moradia por City Tier"),
    html.Div([
        html.Label("Análise:"),
        dcc.RadioItems(
            id='s54-view',
            options=[
                {'label': ' Registros por City Tier',          'value': 'count'},
                {'label': ' Boxplot do Aluguel',               'value': 'boxplot'},
                {'label': ' Violin do Aluguel',                'value': 'violin'},
                {'label': ' Estatísticas (Média/Mediana/DP)',  'value': 'stats'},
                {'label': ' Rent/Income por City Tier',        'value': 'ratio'},
            ],
            value='count',
            inputStyle={'marginRight': '6px'},
            labelStyle={'display': 'block', 'marginBottom': '6px'},
        ),
    ], style={'marginBottom': '15px'}),
    dcc.Graph(id='s54-graph'),
])


def register_callbacks(app):
    @app.callback(
        Output('s54-graph', 'figure'),
        Input('filtered-data-store', 'data'),
        Input('s54-view', 'value'),
    )
    def update_s54(stored_data, view):
        if not stored_data:
            return {}

        dff = pd.DataFrame(stored_data).dropna(subset=['City_Tier', 'Rent']).copy()

        if view == 'count':
            counts = dff['City_Tier'].value_counts().reindex(TIERS).reset_index()
            counts.columns = ['City_Tier', 'Registros']
            fig = px.bar(counts, x='City_Tier', y='Registros', color='City_Tier',
                         text='Registros', title='Registros por City Tier',
                         labels=LABELS, category_orders={'City_Tier': TIERS})
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False)

        elif view == 'boxplot':
            fig = px.box(dff, x='City_Tier', y='Rent', color='City_Tier',
                         title='Distribuição do Aluguel por City Tier',
                         labels=LABELS, category_orders={'City_Tier': TIERS})
            fig.update_layout(showlegend=False)

        elif view == 'violin':
            fig = px.violin(dff, x='City_Tier', y='Rent', color='City_Tier',
                            box=True, points=False,
                            title='Densidade do Aluguel por City Tier',
                            labels=LABELS, category_orders={'City_Tier': TIERS})
            fig.update_layout(showlegend=False)

        elif view == 'stats':
            stats = (dff.groupby('City_Tier')['Rent']
                     .agg(Média='mean', Mediana='median', Desvio='std')
                     .reindex(TIERS).reset_index()
                     .melt(id_vars='City_Tier', var_name='Estatística', value_name='Valor'))
            fig = px.bar(stats, x='City_Tier', y='Valor', color='Estatística',
                         barmode='group',
                         text=stats['Valor'].apply(lambda v: f'${v:,.0f}'),
                         title='Estatísticas do Aluguel por City Tier',
                         labels={'City_Tier': 'City Tier', 'Valor': 'Valor ($)'},
                         category_orders={'City_Tier': TIERS})
            fig.update_traces(textposition='outside')

        else:  # ratio
            avg = dff.groupby('City_Tier')['Rent_Income_Ratio'].mean().reindex(TIERS).reset_index()
            fig = px.bar(avg, x='City_Tier', y='Rent_Income_Ratio', color='City_Tier',
                         text=avg['Rent_Income_Ratio'].apply(lambda v: f'{v:.0%}'),
                         title='Rent/Income por City Tier',
                         labels={'City_Tier': 'City Tier', 'Rent_Income_Ratio': '% da Renda'},
                         category_orders={'City_Tier': TIERS})
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False)
            fig.add_hline(y=0.30, line_dash='dash', line_color='red',
                          annotation_text='Limite 30%', annotation_position='top right')

        fig.update_layout(height=460, margin=dict(t=50, b=50, l=60, r=20))
        return fig
