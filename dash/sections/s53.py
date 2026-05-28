from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Rótulos limpos para exibição (sem underscores)
AGE5  = ['18-25', '26-35', '36-45', '46-55', '55+']
TIERS = ['Tier 1', 'Tier 2', 'Tier 3']
OCCS  = ['Self Employed', 'Retired', 'Student', 'Professional']

layout = html.Div([
    html.H2("5.3 – Condicionantes Financeiros ao Longo da Vida", 
            style={'marginBottom': '20px', 'color': '#1D1252', 'fontSize': '22px'}),
    
    html.Div([
        html.Label("Eixo de Análise:", style={'fontWeight': '600', 'marginRight': '16px', 'display': 'block', 'marginBottom': '8px'}),
        dcc.RadioItems(
            id='s53-view',
            options=[
                {'label': ' Renda Média por Idade',             'value': 'bar_age'},
                {'label': ' Renda por Ocupação',                'value': 'box_occ'},
                {'label': ' Gastos Fixos por City Tier',        'value': 'bar_fixed'},
                {'label': ' Mapa de Correlação',                'value': 'heatmap_corr'},
                {'label': ' Renda Livre (Idade × Dependentes)', 'value': 'heatmap_disp'},
                {'label': ' Trajetória de Renda por Ocupação',  'value': 'line_traj'},
            ],
            value='bar_age',
            style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))', 'gap': '12px'},
            inputStyle={'marginRight': '6px'},
            labelStyle={'color': '#333', 'cursor': 'pointer'},
        ),
    ], style={'marginBottom': '24px', 'backgroundColor': '#f8f9fa', 'padding': '16px', 'borderRadius': '8px', 'border': '1px solid #eaeaea'}),

    dcc.Loading(
        type='dot',
        color='#1D1252',
        children=dcc.Graph(id='s53-graph', config={'displayModeBar': False})
    ),

    html.Div([
        html.Strong("Insight Analítico: ", style={'color': '#1D1252'}),
        html.Span(id='s53-insight')
    ], style={
        'padding': '16px', 
        'backgroundColor': '#f8f9fa', 
        'borderLeft': '4px solid #1D1252',
        'borderRadius': '4px', 
        'marginTop': '16px', 
        'fontSize': '14px'
    }),
])

def _cat(series, order):
    return pd.Categorical(series, categories=order, ordered=True)

def register_callbacks(app):
    @app.callback(
        Output('s53-graph', 'figure'),
        Output('s53-insight', 'children'),
        Input('filtered-data-store', 'data'),
        Input('s53-view', 'value'),
    )
    def update_s53(stored_data, view):
        if not stored_data:
            return {}, "Nenhum dado disponível após os filtros."

        dff = pd.DataFrame(stored_data).copy()

        # Limpeza visual (apenas se a coluna existir)
        if 'City_Tier' in dff.columns:
            dff['City_Tier'] = dff['City_Tier'].astype(str).str.replace('_', ' ')
        if 'Occupation' in dff.columns:
            dff['Occupation'] = dff['Occupation'].astype(str).str.replace('_', ' ')

        brand_color = '#1D1252'
        seq_colors = ['#1D1252', '#594CA3', '#7B6FCD', '#A095F8', '#CBBFFF']
        fig = None
        insight = ""

        if view == 'bar_age':
            if not all(col in dff.columns for col in ['Age_Group_5', 'Income']): return {}, "Erro: Colunas ausentes."
            df_plot = dff.dropna(subset=['Age_Group_5', 'Income']).copy()
            df_plot['Age_Group_5'] = _cat(df_plot['Age_Group_5'], AGE5)
            avg = df_plot.groupby('Age_Group_5', observed=True)['Income'].mean().reset_index()
            
            fig = px.bar(avg, x='Age_Group_5', y='Income', 
                         text=avg['Income'].apply(lambda v: f'${v:,.0f}'),
                         labels={'Age_Group_5': 'Faixa Etária', 'Income': 'Renda Média ($)'},
                         category_orders={'Age_Group_5': AGE5})
            fig.update_traces(marker_color=brand_color, textposition='outside', textfont=dict(weight='bold'))
            
            if not avg.empty:
                max_age = avg.loc[avg['Income'].idxmax()]['Age_Group_5']
                insight = html.Span([f"O pico de geração de renda ocorre na faixa etária ", html.Strong(max_age), ". Isso baliza o momento ideal para maximização de aportes."])

        elif view == 'box_occ':
            if not all(col in dff.columns for col in ['Occupation', 'Income']): return {}, "Erro: Colunas ausentes."
            df_plot = dff.dropna(subset=['Occupation', 'Income']).copy()
            fig = px.box(df_plot, x='Occupation', y='Income', color='Occupation',
                         color_discrete_sequence=seq_colors,
                         labels={'Occupation': 'Ocupação', 'Income': 'Renda ($)'},
                         category_orders={'Occupation': OCCS})
            insight = "Observe a dispersão: Profissionais e autônomos tendem a ter maior variação (outliers), indicando tetos de ganho mais elásticos."

        elif view == 'bar_fixed':
            if not all(col in dff.columns for col in ['City_Tier', 'Fixed_Ratio']): return {}, "Erro: Colunas ausentes."
            df_plot = dff.dropna(subset=['City_Tier', 'Fixed_Ratio']).copy()
            avg = df_plot.groupby('City_Tier')['Fixed_Ratio'].mean().reindex(TIERS).reset_index()
            fig = px.bar(avg, x='City_Tier', y='Fixed_Ratio',
                         text=avg['Fixed_Ratio'].apply(lambda v: f'{v:.1%}'),
                         labels={'City_Tier': 'Classificação da Cidade', 'Fixed_Ratio': 'Custos Fixos / Renda (%)'},
                         category_orders={'City_Tier': TIERS})
            fig.update_traces(marker_color=brand_color, textposition='outside', textfont=dict(weight='bold'))
            fig.update_yaxes(tickformat=".0%")
            
            if not avg['Fixed_Ratio'].isna().all():
                max_tier = avg.loc[avg['Fixed_Ratio'].idxmax()]['City_Tier']
                insight = html.Span([f"O comprometimento da renda com custos essenciais é mais severo no ", html.Strong(max_tier), ", apontando menor margem de segurança."])

        elif view == 'heatmap_corr':
            # Filtra apenas as colunas numéricas que realmente estão no banco para evitar KeyError
            cols = ['Age', 'Dependents', 'Income', 'Rent', 'Loan_Repayment', 'Insurance', 'Fixed_Ratio', 'Disposable_Income', 'Savings_Rate']
            cols = [c for c in cols if c in dff.columns]
            corr = dff[cols].corr(numeric_only=True).round(2)
            
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='Purples', zmin=-1, zmax=1, aspect="auto")
            fig.update_xaxes(tickangle=-45)
            insight = "Correlações escuras positivas indicam variáveis que crescem juntas. Na modelagem preditiva (M1), devemos evitar usá-las simultaneamente para prevenir multicolinearidade."

        elif view == 'heatmap_disp':
            if not all(col in dff.columns for col in ['Age_Group_5', 'Dependents', 'Disposable_Income']): return {}, "Erro: Colunas ausentes."
            df_plot = dff.dropna(subset=['Age_Group_5', 'Dependents', 'Disposable_Income']).copy()
            df_plot['Age_Group_5'] = _cat(df_plot['Age_Group_5'], AGE5)
            pivot = (df_plot.groupby(['Age_Group_5', 'Dependents'], observed=True)['Disposable_Income']
                     .mean().unstack('Dependents'))
            pivot.index   = pivot.index.astype(str)
            pivot.columns = pivot.columns.astype(str)
            fig = px.imshow(pivot, text_auto='.0f', color_continuous_scale='Purples',
                            labels={'x': 'Nº de Dependentes', 'y': 'Faixa Etária', 'color': 'Renda Livre ($)'}, aspect="auto")
            insight = "Quanto mais à direita (dependentes) e abaixo (menor idade), menor tende a ser a renda disponível livre para investimentos."

        elif view == 'line_traj':
            if not all(col in dff.columns for col in ['Age_Group_5', 'Occupation', 'Income']): return {}, "Erro: Colunas ausentes."
            df_plot = dff.dropna(subset=['Age_Group_5', 'Occupation', 'Income']).copy()
            df_plot['Age_Group_5'] = _cat(df_plot['Age_Group_5'], AGE5)
            traj = (df_plot.groupby(['Age_Group_5', 'Occupation'], observed=True)['Income']
                    .median().reset_index())
            fig = px.line(traj, x='Age_Group_5', y='Income', color='Occupation',
                          markers=True, color_discrete_sequence=seq_colors,
                          labels={'Age_Group_5': 'Faixa Etária', 'Income': 'Renda Mediana ($)'},
                          category_orders={'Age_Group_5': AGE5, 'Occupation': OCCS})
            fig.update_traces(line=dict(width=3), marker=dict(size=8))
            insight = "A trajetória revela o 'ciclo de vida' financeiro por ocupação. Quedas abruptas na curva podem indicar transições de carreira ou redução na capacidade laborativa."

        # Aplicação do Design System no Plotly
        if fig:
            fig.update_layout(
                height=450, 
                plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(t=20, b=40, l=60, r=20),
                font=dict(family="Segoe UI, Roboto, sans-serif", color="#333"),
                title=None, 
                showlegend=(view in ['box_occ', 'line_traj']),
                legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_xaxes(showline=True, linewidth=1, linecolor='#e0e0e0', showgrid=False)
            fig.update_yaxes(showline=False, showgrid=True, gridwidth=1, gridcolor='#f0f0f0')

        return fig, insight