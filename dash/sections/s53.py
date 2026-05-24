from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

AGE5  = ['18-25', '26-35', '36-45', '46-55', '55+']
TIERS = ['Tier_1', 'Tier_2', 'Tier_3']
OCCS  = ['Self_Employed', 'Retired', 'Student', 'Professional']

layout = html.Div([
    html.H2("5.3 - Condicionantes Financeiros ao Longo da Vida"),
    html.Div([
        html.Label("Análise:"),
        dcc.RadioItems(
            id='s53-view',
            options=[
                {'label': ' Renda Média por Faixa Etária',                 'value': 'bar_age'},
                {'label': ' Distribuição de Renda por Ocupação',            'value': 'box_occ'},
                {'label': ' Gastos Fixos por City Tier',                    'value': 'bar_fixed'},
                {'label': ' Mapa de Correlação',                            'value': 'heatmap_corr'},
                {'label': ' Renda Disponível (Faixa Etária × Dependentes)', 'value': 'heatmap_disp'},
                {'label': ' Trajetória de Renda por Ocupação',              'value': 'line_traj'},
            ],
            value='bar_age',
            inputStyle={'marginRight': '6px'},
            labelStyle={'display': 'block', 'marginBottom': '6px'},
        ),
    ], style={'marginBottom': '15px'}),
    dcc.Graph(id='s53-graph'),
])


def _cat(series, order):
    return pd.Categorical(series, categories=order, ordered=True)


def register_callbacks(app):
    @app.callback(
        Output('s53-graph', 'figure'),
        Input('filtered-data-store', 'data'),
        Input('s53-view', 'value'),
    )
    def update_s53(stored_data, view):
        if not stored_data:
            return {}

        dff = pd.DataFrame(stored_data).copy()

        if view == 'bar_age':
            dff = dff.dropna(subset=['Age_Group_5', 'Income'])
            dff['Age_Group_5'] = _cat(dff['Age_Group_5'], AGE5)
            avg = dff.groupby('Age_Group_5', observed=True)['Income'].mean().reset_index()
            fig = px.bar(avg, x='Age_Group_5', y='Income', color='Age_Group_5',
                         text=avg['Income'].apply(lambda v: f'${v:,.0f}'),
                         title='Renda Média por Faixa Etária',
                         labels={'Age_Group_5': 'Faixa Etária', 'Income': 'Renda Média ($)'},
                         category_orders={'Age_Group_5': AGE5})
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False)

        elif view == 'box_occ':
            dff = dff.dropna(subset=['Occupation', 'Income'])
            fig = px.box(dff, x='Occupation', y='Income', color='Occupation',
                         title='Distribuição de Renda por Ocupação',
                         labels={'Occupation': 'Ocupação', 'Income': 'Renda ($)'},
                         category_orders={'Occupation': OCCS})
            fig.update_layout(showlegend=False)

        elif view == 'bar_fixed':
            dff = dff.dropna(subset=['City_Tier', 'Fixed_Ratio'])
            avg = dff.groupby('City_Tier')['Fixed_Ratio'].mean().reindex(TIERS).reset_index()
            fig = px.bar(avg, x='City_Tier', y='Fixed_Ratio', color='City_Tier',
                         text=avg['Fixed_Ratio'].apply(lambda v: f'{v:.0%}'),
                         title='Comprometimento da Renda com Gastos Fixos por City Tier',
                         labels={'City_Tier': 'City Tier', 'Fixed_Ratio': 'Proporção da Renda'},
                         category_orders={'City_Tier': TIERS})
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False)

        elif view == 'heatmap_corr':
            cols = ['Age', 'Dependents', 'Income', 'Rent', 'Loan_Repayment',
                    'Insurance', 'Fixed_Ratio', 'Disposable_Income', 'Savings_Rate']
            corr = dff[cols].corr(numeric_only=True).round(2)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1, title='Mapa de Correlação')

        elif view == 'heatmap_disp':
            dff = dff.dropna(subset=['Age_Group_5', 'Dependents', 'Disposable_Income'])
            dff['Age_Group_5'] = _cat(dff['Age_Group_5'], AGE5)
            pivot = (dff.groupby(['Age_Group_5', 'Dependents'], observed=True)['Disposable_Income']
                     .mean().unstack('Dependents'))
            pivot.index   = pivot.index.astype(str)
            pivot.columns = pivot.columns.astype(str)
            fig = px.imshow(pivot, text_auto='.0f', color_continuous_scale='Greens',
                            title='Renda Disponível Média por Faixa Etária × Nº de Dependentes',
                            labels={'x': 'Nº de Dependentes', 'y': 'Faixa Etária', 'color': '$'})

        else:  # line_traj
            dff = dff.dropna(subset=['Age_Group_5', 'Occupation', 'Income'])
            dff['Age_Group_5'] = _cat(dff['Age_Group_5'], AGE5)
            traj = (dff.groupby(['Age_Group_5', 'Occupation'], observed=True)['Income']
                    .median().reset_index())
            fig = px.line(traj, x='Age_Group_5', y='Income', color='Occupation',
                          markers=True,
                          title='Trajetória de Renda Mediana por Ocupação',
                          labels={'Age_Group_5': 'Faixa Etária', 'Income': 'Renda Mediana ($)'},
                          category_orders={'Age_Group_5': AGE5, 'Occupation': OCCS})

        fig.update_layout(height=480, margin=dict(t=50, b=50, l=60, r=20))
        return fig
