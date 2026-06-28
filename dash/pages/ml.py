import dash
from dash import dcc, html, Input, Output, State, MATCH, ALL, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import joblib
import os

# ─── Caminhos ────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(ROOT_DIR, '..', 'modelos', 'mecanismo_vulnerabilidade', 'modelo_regressao_logistica_seguro.pkl')
DATA_PATH         = os.path.join(ROOT_DIR, 'dataset', 'data.parquet')
CLUSTER_DATA_PATH = os.path.join(ROOT_DIR, '..', 'modelos', 'modelo_clusterizacao', 'data.parquet')

FEATURES = [
    'Age', 'Dependents', 'Rent_Ratio', 'Healthcare_Ratio', 'Education_Ratio',
    'Groceries_Ratio', 'Transport_Ratio', 'Utilities_Ratio', 'Insurance_Ratio',
]

# ─── Design tokens (idênticos ao resto do projeto) ───────────────────────────
ROXO        = '#1D1252'
BRANCO      = '#FFFFFF'
TEXT_MAIN   = '#111827'
TEXT_MUTED  = '#555555'
BORDER      = '#E5E7EB'
SUCCESS     = '#2e7d32'
ERROR       = '#E54B4B'
BG_PAGE     = '#F9FAFB'

CARD = {
    'backgroundColor': BRANCO,
    'borderRadius': '12px',
    'padding': '32px',
    'marginBottom': '32px',
    'border': f'1px solid {BORDER}',
    'boxShadow': '0 4px 20px rgba(0,0,0,0.04)',
}

P_STYLE = {
    'color': TEXT_MUTED,
    'marginBottom': '20px',
    'lineHeight': '1.75',
    'fontSize': '15px',
    'textAlign': 'justify',
}

H2 = {'color': TEXT_MAIN, 'fontSize': '22px', 'marginBottom': '16px', 'marginTop': '0'}
H3 = {'color': ROXO,      'fontSize': '17px', 'marginBottom': '12px', 'marginTop': '24px'}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ip_field(field_id, label, default, hint=None):
    """Campo numérico com botões − e + customizados."""
    _btn = {
        'width': '34px', 'height': '38px', 'border': f'1px solid {BORDER}',
        'borderRadius': '6px', 'cursor': 'pointer', 'background': '#f0eeff',
        'color': ROXO, 'fontWeight': '700', 'fontSize': '18px', 'lineHeight': '1',
        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
        'flexShrink': '0', 'padding': '0',
    }
    children = [
        html.Label(label, style={
            'fontSize': '12px', 'color': TEXT_MUTED, 'fontWeight': '600',
            'display': 'block', 'marginBottom': '5px',
        }),
        html.Div([
            html.Button('−', id={'type': 'ip-minus', 'id': field_id}, n_clicks=0, style=_btn),
            dcc.Input(
                id={'type': 'ip-num', 'id': field_id},
                type='text', value=str(default),
                inputMode='numeric',
                style={
                    'flex': '1', 'padding': '9px 10px', 'borderRadius': '6px',
                    'border': f'1px solid {BORDER}', 'fontSize': '14px', 'color': TEXT_MAIN,
                    'boxSizing': 'border-box', 'outline': 'none', 'textAlign': 'right', 'minWidth': '0',
                },
            ),
            html.Button('+', id={'type': 'ip-plus', 'id': field_id}, n_clicks=0, style=_btn),
        ], style={'display': 'flex', 'gap': '6px', 'alignItems': 'center'}),
    ]
    if hint:
        children.append(html.Div(hint, style={
            'fontSize': '11px', 'color': TEXT_MUTED, 'marginTop': '4px', 'lineHeight': '1.4',
        }))
    return html.Div(children, style={'marginBottom': '14px'})


def badge(text, color=ROXO):
    return html.Span(text, style={
        'backgroundColor': color, 'color': BRANCO,
        'borderRadius': '20px', 'padding': '4px 14px',
        'fontSize': '12px', 'fontWeight': '700',
        'display': 'inline-block', 'marginRight': '8px',
    })

def info_box(children, border_color=ROXO):
    return html.Div(children, style={
        'borderLeft': f'4px solid {border_color}',
        'backgroundColor': 'rgba(29,18,82,0.04)',
        'padding': '16px 20px',
        'borderRadius': '0 8px 8px 0',
        'marginBottom': '16px',
        'fontSize': '14px',
        'color': TEXT_MAIN,
        'lineHeight': '1.7',
    })

def metric_card(label, value, color=ROXO):
    return html.Div([
        html.Div(label, style={'color': '#666', 'fontSize': '12px', 'fontWeight': '600',
                               'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
        html.Div(value, style={'color': color, 'fontSize': '26px', 'fontWeight': 'bold', 'marginTop': '6px'}),
    ], style={
        'backgroundColor': BRANCO, 'padding': '20px', 'borderRadius': '10px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.06)', 'borderLeft': f'4px solid {color}',
    })

def html_table(headers, rows, highlight_row=None):
    """Tabela HTML estilizada, com suporte a highlight_row (índice 0-based)."""
    def th(text):
        return html.Th(text, style={
            'backgroundColor': ROXO, 'color': BRANCO,
            'padding': '10px 14px', 'textAlign': 'center',
            'fontSize': '13px', 'fontWeight': '600',
        })

    def td(text, bold=False, color=None):
        return html.Td(text, style={
            'padding': '9px 14px', 'textAlign': 'center',
            'fontSize': '13px', 'color': color or TEXT_MAIN,
            'fontWeight': '700' if bold else '400',
        })

    body_rows = []
    for i, row in enumerate(rows):
        bg = 'rgba(29,18,82,0.07)' if i == highlight_row else ('rgba(0,0,0,0.01)' if i % 2 else BRANCO)
        cells = [td(c, bold=(i == highlight_row)) for c in row]
        body_rows.append(html.Tr(cells, style={'backgroundColor': bg}))

    return html.Table([
        html.Thead(html.Tr([th(h) for h in headers])),
        html.Tbody(body_rows),
    ], style={
        'width': '100%', 'borderCollapse': 'collapse',
        'borderRadius': '8px', 'overflow': 'hidden',
        'boxShadow': '0 1px 4px rgba(0,0,0,0.06)',
    })

def section_img(src, caption, max_width='680px'):
    return html.Div([
        html.Img(src=src, style={
            'maxWidth': max_width, 'width': '100%',
            'borderRadius': '8px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.1)',
            'display': 'block', 'margin': '0 auto',
        }),
        html.P(caption, style={
            'textAlign': 'center', 'fontSize': '13px',
            'color': TEXT_MUTED, 'marginTop': '10px', 'fontStyle': 'italic',
        }),
    ], style={'marginTop': '24px', 'marginBottom': '24px'})


# ─── Pré-computação dos gráficos estáticos do modelo ──────────────────────────
# Executado uma única vez no startup do servidor (sem sklearn em runtime).
def _build_static_charts():
    """
    Gera os 5 gráficos interativos do modelo usando apenas pandas/numpy/plotly.
    Retorna: (corr_fig, cm_fig, shap_fig, pareto_fig, cv_fig)
    Foco: classe Seguro (0) — identificação de bons clientes para crédito.
    """
    _placeholder = go.Figure().update_layout(
        paper_bgcolor='white', height=360,
        annotations=[dict(text='Dados do modelo não disponíveis',
                          x=0.5, y=0.5, xref='paper', yref='paper',
                          showarrow=False, font=dict(size=13, color='#aaa'))],
        margin=dict(t=20, b=20, l=20, r=20),
    )
    try:
        # ── Carregar modelo ──────────────────────────────────────────────────
        p      = joblib.load(MODEL_PATH)
        coef   = np.array(p['coef']).ravel()        # (9,)
        intcpt = float(np.array(p['intercept']).ravel()[0])

        # ── Carregar e preparar dados ─────────────────────────────────────
        try:
            df = pd.read_parquet(DATA_PATH, engine='fastparquet')
        except Exception:
            df = pd.read_parquet(DATA_PATH)
        df = df[df['Desired_Savings'] > 0].reset_index(drop=True)

        # Construir variável alvo
        df['perc_nao_essenciais'] = (df['Eating_Out'] + df['Entertainment']) / df['Income']
        df['perc_emprestimo']     = df['Loan_Repayment'] / df['Income']
        df['Vulnerable'] = (
            (df['perc_emprestimo'] > 0.10).astype(int) +
            (df['perc_nao_essenciais'] > 0.085).astype(int) +
            ((df['Disposable_Income'] - df['Desired_Savings']) / df['Income'] < 0.10).astype(int) +
            (df['Potential_Savings_Groceries'] / df['Income'] > 0.08).astype(int) >= 2
        ).astype(int)

        # Engenharia de features (rácios)
        df['Rent_Ratio']       = df['Rent']       / df['Income']
        df['Healthcare_Ratio'] = df['Healthcare'] / df['Income']
        df['Education_Ratio']  = df['Education']  / df['Income']
        df['Groceries_Ratio']  = df['Groceries']  / df['Income']
        df['Transport_Ratio']  = df['Transport']  / df['Income']
        df['Utilities_Ratio']  = df['Utilities']  / df['Income']
        df['Insurance_Ratio']  = df['Insurance']  / df['Income']

        X   = df[FEATURES]
        feat_labels = [f.replace('_', ' ') for f in FEATURES]

        # ── 1. Matriz de Correlação ───────────────────────────────────────
        corr = X.corr().round(2)
        corr_fig = px.imshow(
            corr.values,
            x=feat_labels, y=feat_labels,
            text_auto='.2f',
            color_continuous_scale='RdYlGn',
            range_color=[-1, 1],
            title='Gráfico 1 — Matriz de Correlação das 9 Features Preditoras (Rácios)',
        )
        corr_fig.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            height=500, margin=dict(t=60, b=100, l=150, r=40),
            font=dict(size=11, family='Segoe UI'),
            title_font_size=13,
            coloraxis_colorbar=dict(title='r'),
        )
        corr_fig.update_xaxes(tickangle=-40)
        corr_fig.update_traces(
            hovertemplate='<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:.2f}<extra></extra>'
        )

        # ── 2. Matriz de Confusão (valores exatos do notebook) ────────────
        # sklearn confusion_matrix(y_test, y_pred_lr), train_test_split(random_state=42)
        # Seguro=0, Vulnerável=1  →  cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
        tn, fp, fn, tp = 2522, 636, 532, 288
        recall_seg = tn / (tn + fp)   # 79,9 %
        auc_roc    = p.get('auc_roc', '0.6529')

        lbs    = ['Seguro', 'Vulnerável']
        cm_z   = [[tn, fp], [fn, tp]]
        cm_txt = [[f'{tn:,}', f'{fp:,}'], [f'{fn:,}', f'{tp:,}']]

        cm_fig = go.Figure(go.Heatmap(
            z=cm_z, x=lbs, y=lbs,
            text=cm_txt, texttemplate='<b>%{text}</b>',
            colorscale='Blues', showscale=True,
            hovertemplate='Real: %{y}<br>Previsto: %{x}<br>Contagem: %{text}<extra></extra>',
        ))
        cm_fig.update_layout(
            title='Figura 17 — Matriz de Confusão, Regressão Logística (Foco: Seguro)',
            xaxis_title='Predito',
            yaxis=dict(title='Real', autorange='reversed'),
            paper_bgcolor='white', plot_bgcolor='white',
            height=420, font=dict(size=12, family='Segoe UI'),
            title_font_size=13,
            margin=dict(t=60, b=90, l=120, r=40),
            annotations=[dict(
                text=f'Recall Seguro: {recall_seg*100:.1f}%  |  '
                     f'Seguro corretos: {tn:,} de {tn+fp:,}  |  '
                     f'AUC-ROC: {auc_roc}',
                x=0.5, y=-0.28, xref='paper', yref='paper',
                showarrow=False, font=dict(size=12, color=TEXT_MUTED),
            )],
        )

        # ── 3. SHAP Importance (perspectiva Seguro, valores exatos do notebook)
        # shap.LinearExplainer(modelo_lr, X_train).shap_values(X_test), sv = -shap, mask y_test==0
        shap_exact = {
            'Healthcare Ratio': 0.004637,
            'Transport Ratio':  0.013670,
            'Age':              0.015600,
            'Insurance Ratio':  0.016555,
            'Dependents':       0.030998,
            'Utilities Ratio':  0.035733,
            'Groceries Ratio':  0.057591,
            'Education Ratio':  0.185741,
            'Rent Ratio':       0.284150,
        }
        shap_df = pd.DataFrame(list(shap_exact.items()), columns=['feature', 'importancia'])
        shap_df = shap_df.sort_values('importancia', ascending=True).reset_index(drop=True)

        shap_fig = go.Figure(go.Bar(
            x=shap_df['importancia'],
            y=shap_df['feature'],
            orientation='h',
            marker_color=SUCCESS,
            hovertemplate='<b>%{y}</b><br>|SHAP| médio: %{x:.4f}<extra></extra>',
        ))
        shap_fig.update_layout(
            title='Figura 18 — SHAP: Importância das Features para Seguro (Regressão Logística)',
            xaxis_title='Importância média (|SHAP value|) — perspectiva Seguro',
            paper_bgcolor='white', plot_bgcolor='white',
            height=420, font=dict(size=12, family='Segoe UI'),
            title_font_size=13,
            margin=dict(t=60, b=40, l=160, r=40),
        )
        shap_fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')

        # ── 4. Princípio de Pareto (valores exatos do notebook) ─────────
        pareto_exact = [
            ('Rent Ratio',       44.076),
            ('Education Ratio',  28.812),
            ('Groceries Ratio',   8.933),
            ('Utilities Ratio',   5.543),
            ('Dependents',        4.808),
            ('Insurance Ratio',   2.568),
            ('Age',               2.420),
            ('Transport Ratio',   2.121),
            ('Healthcare Ratio',  0.719),
        ]
        pareto_df = pd.DataFrame(pareto_exact, columns=['feature', 'pct'])
        pareto_df['cumulativo'] = pareto_df['pct'].cumsum()

        pareto_fig = go.Figure()
        pareto_fig.add_trace(go.Bar(
            x=pareto_df['feature'], y=pareto_df['pct'],
            name='Importância %', marker_color=SUCCESS,
            hovertemplate='<b>%{x}</b><br>Importância: %{y:.1f}%<extra></extra>',
        ))
        pareto_fig.add_trace(go.Scatter(
            x=pareto_df['feature'], y=pareto_df['cumulativo'],
            mode='lines+markers', name='Cumulativo %',
            yaxis='y2',
            line=dict(color=ERROR, width=2.5),
            marker=dict(size=6, color=ERROR),
            hovertemplate='<b>%{x}</b><br>Cumulativo: %{y:.1f}%<extra></extra>',
        ))
        pareto_fig.add_trace(go.Scatter(
            x=pareto_df['feature'].tolist(), y=[80] * len(pareto_df),
            mode='lines', name='80%', yaxis='y2',
            line=dict(color='gray', dash='dash', width=1.5),
            hoverinfo='skip', showlegend=True,
        ))
        pareto_fig.update_layout(
            title='Figura 19 — Princípio de Pareto: Importância das Features para Seguro',
            yaxis=dict(title='Importância individual (%)', showgrid=True, gridcolor='#f0f0f0'),
            yaxis2=dict(title='Cumulativo (%)', overlaying='y', side='right', range=[0, 110]),
            plot_bgcolor='white', paper_bgcolor='white',
            height=440, font=dict(size=11, family='Segoe UI'),
            title_font_size=13,
            legend=dict(orientation='h', y=-0.28, x=0.3),
            margin=dict(t=60, b=90, l=60, r=70),
        )
        pareto_fig.update_xaxes(tickangle=-30)

        # ── 5. Cross-Validation (tabela visual) ───────────────────────────
        cv_data = {
            'Métrica':  ['Precisão', 'Recall', 'F1-Score', 'Acurácia'],
            'Média':    ['0,827',    '0,794',  '0,810',    '0,705'],
            'Desvio':   ['+/- 0,003','+/- 0,009','+/- 0,005','+/- 0,006'],
        }
        cv_df = pd.DataFrame(cv_data)

        cv_fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Métrica</b>', '<b>Média</b>', '<b>Desvio Padrão</b>'],
                fill_color=ROXO,
                font=dict(color='white', size=13, family='Segoe UI'),
                align='center',
                height=36,
            ),
            cells=dict(
                values=[cv_df['Métrica'], cv_df['Média'], cv_df['Desvio']],
                fill_color=[['white', '#e3f2fd'] * 4],
                font=dict(color=TEXT_MAIN, size=13, family='Segoe UI'),
                align='center',
                height=32,
            ),
        )])
        cv_fig.update_layout(
            title='Cross-Validation — Regressão Logística (5 Folds) — Foco: Seguro',
            title_font_size=13,
            paper_bgcolor='white',
            height=280,
            margin=dict(t=50, b=20, l=20, r=20),
            font=dict(family='Segoe UI'),
        )

        return corr_fig, cm_fig, shap_fig, pareto_fig, cv_fig

    except Exception as e:
        print(f'[_build_static_charts error] {e}')
        return _placeholder, _placeholder, _placeholder, _placeholder, _placeholder


_CORR_FIG, _CM_FIG, _SHAP_FIG, _PARETO_FIG, _CV_FIG = _build_static_charts()


# ─── Pré-computação dos gráficos interativos do modelo de clusterização ──────
def _build_cluster_charts():
    """
    Gera os 8 gráficos interativos do modelo K-Means++.
    Retorna: (corr_fig, pca_var_fig, kdist_fig, elbow_sil_fig,
              silh_diag_fig, pca2d_fig, shap_fig, pareto_fig)
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples
    from sklearn.neighbors import NearestNeighbors
    from sklearn.ensemble import RandomForestClassifier

    _ph = go.Figure().update_layout(
        paper_bgcolor='white', height=400,
        annotations=[dict(text='Dados de cluster não disponíveis', x=0.5, y=0.5,
                          xref='paper', yref='paper', showarrow=False,
                          font=dict(size=13, color='#aaa'))],
        margin=dict(t=20, b=20, l=20, r=20),
    )
    try:
        # ── Feature engineering (idêntico ao notebook) ────────────────────────
        df = pd.read_parquet(CLUSTER_DATA_PATH)
        city_map = {'Tier_1': 3, 'Tier_2': 2, 'Tier_3': 1}
        df['City_Tier_enc'] = df['City_Tier'].map(city_map)

        df_m = pd.DataFrame({
            'Rent_pct':            df['Rent']             / df['Income'],
            'Loan_Repayment_pct':  df['Loan_Repayment']   / df['Income'],
            'Disposable_pct':      df['Disposable_Income'] / df['Income'],
            # Desired_Savings_Percentage é uma coluna de percentual (0–100); /100 → fração
            'Desired_Savings_pct': df['Desired_Savings_Percentage'] / 100,
            'Age':                 df['Age'],
            'Dependents':          df['Dependents'],
            # Gap = meta de poupança (fração) − renda disponível (fração)
            'Gap_Poupanca':        (df['Desired_Savings_Percentage'] / 100) - (df['Disposable_Income'] / df['Income']),
            'City_Tier_enc':       df['City_Tier_enc'],
            'Gastos_Consumo_pct':  (df['Eating_Out'] + df['Entertainment'] + df['Groceries']) / df['Income'],
            # Gastos_Fixos = Insurance + Transport + Healthcare (sem Utilities — igual ao notebook)
            'Gastos_Fixos_pct':    (df['Insurance'] + df['Transport'] + df['Healthcare']) / df['Income'],
        })

        scaler   = RobustScaler()
        df_norm  = scaler.fit_transform(df_m)
        feat_names = list(df_m.columns)
        feat_lbl   = [n.replace('_pct', ' %').replace('_', ' ') for n in feat_names]

        # ── 1. Correlação ─────────────────────────────────────────────────────
        # Usa os nomes originais das colunas (igual ao notebook: sns.heatmap com df_modelo.corr())
        corr     = df_m.corr().round(2)
        col_names = list(df_m.columns)   # Rent_pct, Loan_Repayment_pct, etc.
        corr_fig = px.imshow(
            corr.values, x=col_names, y=col_names,
            text_auto='.2f', color_continuous_scale='RdBu_r',
            range_color=[-1, 1],
            title='Mapa de Correlação — Variáveis Proporcionais',
        )
        corr_fig.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            height=540, margin=dict(t=60, b=130, l=180, r=40),
            font=dict(size=11, family='Segoe UI'), title_font_size=13,
            coloraxis_colorbar=dict(title='r'),
        )
        corr_fig.update_xaxes(tickangle=-45)
        corr_fig.update_traces(
            hovertemplate='<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:.2f}<extra></extra>'
        )

        # ── 2. PCA Variância Acumulada ────────────────────────────────────────
        pca_full = PCA()
        pca_full.fit(df_norm)
        var_cumul = np.cumsum(pca_full.explained_variance_ratio_) * 100
        pcs = list(range(1, len(var_cumul) + 1))

        pca_var_fig = go.Figure()
        pca_var_fig.add_trace(go.Scatter(
            x=pcs, y=var_cumul.tolist(),
            mode='lines+markers',
            line=dict(color='royalblue', width=2),
            marker=dict(symbol='circle', size=8, color='royalblue'),
            hovertemplate='PC%{x}<br>Variância Acumulada: %{y:.1f}%<extra></extra>',
        ))
        pca_var_fig.add_hline(y=85, line=dict(color='red', dash='dash', width=1.5),
                              annotation_text='85%', annotation_position='top right')
        pca_var_fig.add_hline(y=95, line=dict(color='green', dash='dash', width=1.5),
                              annotation_text='95%', annotation_position='top right')
        pca_var_fig.update_layout(
            title='Quantos componentes PCA precisamos?',
            xaxis=dict(title='Número de Componentes', dtick=1),
            yaxis=dict(title='Variância Acumulada (%)'),
            paper_bgcolor='white', plot_bgcolor='white',
            height=440, font=dict(size=12, family='Segoe UI'), title_font_size=13,
            showlegend=False, margin=dict(t=60, b=60, l=70, r=60),
        )

        # ── PCA 6 componentes para clustering (≈85 % variância) ──────────────
        pca6   = PCA(n_components=6, random_state=42)
        df_pca = pca6.fit_transform(df_norm)

        # ── 3. K-Distance (amostra de 5000 pontos para velocidade) ───────────
        rng42    = np.random.RandomState(42)
        sub_idx  = rng42.choice(len(df_pca), 5000, replace=False)
        df_sub   = df_pca[sub_idx]
        nn       = NearestNeighbors(n_neighbors=12).fit(df_sub)
        dists, _ = nn.kneighbors(df_sub)
        kdist    = np.sort(dists[:, 11])

        kdist_fig = go.Figure()
        kdist_fig.add_trace(go.Scatter(
            y=kdist.tolist(), mode='lines',
            line=dict(color=ROXO, width=2),
            hovertemplate='Ponto %{x}<br>Distância: %{y:.4f}<extra></extra>',
            name='K-Distance',
        ))
        kdist_fig.add_hline(y=0.70, line=dict(color='red', dash='dash', width=1.5),
                            annotation_text='ε = 0,70', annotation_position='top right')
        kdist_fig.update_layout(
            title='Gráfico 5 — K-Distance (12-NN) para Determinação do Epsilon (DBSCAN)',
            xaxis_title='Pontos ordenados',
            yaxis_title='Distância ao 12º vizinho mais próximo',
            paper_bgcolor='white', plot_bgcolor='white',
            height=400, font=dict(size=12, family='Segoe UI'), title_font_size=13,
            margin=dict(t=60, b=60, l=60, r=40), showlegend=False,
        )

        # ── 4. Cotovelo + Silhouette Score (valores completos, hardcoded) ─────
        K_range     = list(range(2, 11))
        inertias    = [64057, 56229, 50749, 46342, 42668, 40512, 38788, 37266, 35796]
        silhouettes = [0.2324, 0.2170, 0.2080, 0.2190, 0.1897, 0.1770, 0.1653, 0.1622, 0.1590]

        elbow_sil_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Método do Cotovelo (Inércia)', 'Silhouette Score Médio'],
        )
        elbow_sil_fig.add_trace(go.Scatter(
            x=K_range, y=inertias, mode='lines+markers', name='Inércia',
            line=dict(color='royalblue', width=2),
            marker=dict(symbol='circle', size=8, color='royalblue'),
            hovertemplate='K=%{x}<br>Inércia: %{y:,.0f}<extra></extra>',
        ), row=1, col=1)
        elbow_sil_fig.add_trace(go.Scatter(
            x=K_range, y=silhouettes, mode='lines+markers', name='Silhouette',
            line=dict(color='red', width=2),
            marker=dict(symbol='square', size=8, color='red'),
            hovertemplate='K=%{x}<br>Silhouette: %{y:.4f}<extra></extra>',
        ), row=1, col=2)
        for _c in [1, 2]:
            elbow_sil_fig.add_vline(
                x=5, line=dict(color=ERROR, dash='dash', width=1.5), row=1, col=_c,
            )
        elbow_sil_fig.update_layout(
            title='Gráfico 6 — Seleção de K: Cotovelo e Silhouette Score (K-Means++)',
            paper_bgcolor='white', plot_bgcolor='white',
            height=420, font=dict(size=12, family='Segoe UI'), title_font_size=13,
            margin=dict(t=80, b=60, l=70, r=60), showlegend=False,
        )
        elbow_sil_fig.update_xaxes(title_text='K (clusters)', dtick=1)
        elbow_sil_fig.update_yaxes(title_text='Inércia', row=1, col=1)
        elbow_sil_fig.update_yaxes(title_text='Silhouette Score', row=1, col=2)

        # ── 5. Silhouette Diagram K = 3, 4, 5 (nipy_spectral, gap=10) ───────────
        import matplotlib.cm as _cm
        def _nipy(i, k):
            r, g, b, _ = _cm.nipy_spectral(float(i) / k)
            return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

        silh_diag_fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['K = 3', 'K = 4', 'K = 5'],
        )
        cl_palette = ['steelblue', 'coral', 'green', 'purple', 'orange']
        silh_annots = []

        for col_idx, k in enumerate([3, 4, 5]):
            km  = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            lbl = km.fit_predict(df_pca)
            sv  = silhouette_samples(df_pca, lbl)
            avg = float(sv.mean())
            n_total = len(sv)

            y_lower = 10
            for i in range(k):
                cl_sv   = np.sort(sv[lbl == i])
                n_i     = len(cl_sv)
                y_upper = y_lower + n_i
                y_range = np.arange(y_lower, y_upper)
                color   = _nipy(i, k)

                silh_diag_fig.add_trace(go.Scatter(
                    x=np.concatenate([[0], cl_sv, [0]]).tolist(),
                    y=np.concatenate([[y_lower], y_range, [y_upper]]).tolist(),
                    fill='tozerox', fillcolor=color,
                    line=dict(color=color, width=0.5),
                    name=f'Cluster {i}', opacity=0.75,
                    showlegend=False,
                ), row=1, col=col_idx + 1)

                # Cluster label on the left
                xref = 'x' if col_idx == 0 else f'x{col_idx + 1}'
                yref = 'y' if col_idx == 0 else f'y{col_idx + 1}'
                silh_annots.append(dict(
                    x=-0.05, y=y_lower + 0.5 * n_i,
                    xref=xref, yref=yref,
                    text=str(i), showarrow=False,
                    font=dict(size=11, color='black'),
                ))
                y_lower = y_upper + 10

            # Mean vline + annotation
            silh_diag_fig.add_vline(
                x=avg, line=dict(color='red', dash='dash', width=1.5),
                annotation_text=f'Média: {avg:.3f}',
                annotation_font=dict(size=9, color='red'),
                annotation_position='top right',
                row=1, col=col_idx + 1,
            )

        silh_diag_fig.update_layout(
            title='Silhouette Diagram — Comparação de K',
            annotations=silh_annots,
            paper_bgcolor='white', plot_bgcolor='white',
            height=480, font=dict(size=11, family='Segoe UI'), title_font_size=13,
            margin=dict(t=80, b=60, l=60, r=40),
        )
        silh_diag_fig.update_xaxes(title_text='Silhouette coefficient', range=[-0.1, 1])
        silh_diag_fig.update_yaxes(showticklabels=False)

        # ── Modelo final K = 5 ────────────────────────────────────────────────
        modelo_final = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
        labels_final = modelo_final.fit_predict(df_pca)
        df_m['Cluster'] = labels_final

        # ── 6. 2D PCA Scatter — dados brutos, cor única (sem labels de cluster) ─
        # Replica exatamente o notebook: PCA(n_components=2) no df_normalizado,
        # todos os pontos em steelblue, alpha=0.3, s=5 (marker size ~3px)
        pca2d     = PCA(n_components=2, random_state=42)
        df_pca_2d = pca2d.fit_transform(df_norm)

        pca2d_fig = go.Figure()
        pca2d_fig.add_trace(go.Scatter(
            x=df_pca_2d[:, 0].tolist(),
            y=df_pca_2d[:, 1].tolist(),
            mode='markers',
            marker=dict(color='steelblue', size=3, opacity=0.3),
            hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
            showlegend=False,
        ))
        pca2d_fig.update_layout(
            title='Visualização 2D dos dados (PCA)',
            xaxis_title='Componente Principal 1',
            yaxis_title='Componente Principal 2',
            paper_bgcolor='white', plot_bgcolor='white',
            height=480, font=dict(size=12, family='Segoe UI'), title_font_size=13,
            margin=dict(t=60, b=60, l=60, r=40),
        )

        # ── 7. SHAP Beeswarm (Random Forest Surrogate) ───────────────────────
        X_shap = df_m.drop(columns=['Cluster'])
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        rf.fit(X_shap, labels_final)

        shap_idx = np.random.RandomState(42).choice(len(X_shap), 800, replace=False)
        X_samp   = X_shap.iloc[shap_idx]
        fv_arr   = X_samp.values

        try:
            import shap as _shap
            expl   = _shap.TreeExplainer(rf)
            sv_raw = np.array(expl.shap_values(X_samp))
            if sv_raw.ndim == 3:
                sv_per = sv_raw.mean(axis=0) if sv_raw.shape[0] == 5 else sv_raw.mean(axis=2)
            else:
                sv_per = sv_raw
        except Exception:
            imp    = rf.feature_importances_
            sv_per = np.tile(imp, (len(X_samp), 1))

        importance = np.abs(sv_per).mean(axis=0)
        order      = np.argsort(importance)   # ascending → bottom to top on y-axis

        shap_fig = go.Figure()
        for rank, fi in enumerate(order):
            sv    = sv_per[:, fi]
            fv    = fv_arr[:, fi]
            fnorm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
            jit   = np.random.RandomState(fi * 17).uniform(-0.3, 0.3, len(sv))

            shap_fig.add_trace(go.Scatter(
                x=sv.tolist(), y=(rank + jit).tolist(),
                mode='markers', name=feat_lbl[fi], showlegend=False,
                marker=dict(
                    size=4, color=fnorm.tolist(), colorscale='RdBu_r',
                    cmin=0, cmax=1, opacity=0.65,
                    showscale=(rank == len(order) - 1),
                    colorbar=dict(
                        title='Valor<br>feature',
                        tickvals=[0, 0.5, 1], ticktext=['Baixo', 'Médio', 'Alto'],
                        thickness=12, len=0.55, y=0.5,
                    ) if rank == len(order) - 1 else None,
                ),
                hovertemplate=f'<b>{feat_lbl[fi]}</b><br>SHAP: %{{x:.4f}}<extra></extra>',
            ))

        shap_fig.add_vline(x=0, line=dict(color='gray', width=1, dash='dot'))
        shap_fig.update_layout(
            title='Gráfico 30 — SHAP Beeswarm Global, Random Forest Surrogate (K-Means++)',
            xaxis_title='SHAP Value (impacto na separação de clusters)',
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(order))),
                ticktext=[feat_lbl[fi] for fi in order],
                showgrid=False,
            ),
            paper_bgcolor='white', plot_bgcolor='white',
            height=480, font=dict(size=11, family='Segoe UI'), title_font_size=13,
            margin=dict(t=60, b=60, l=165, r=80),
        )

        # ── 8. Pareto por Cluster (5 subplots com eixo y secundário) ─────────
        cols_pareto = ['Rent_pct', 'Loan_Repayment_pct', 'Gastos_Consumo_pct', 'Gastos_Fixos_pct']
        lbls_pareto = ['Aluguel', 'Empréstimo', 'Consumo', 'Fixos']
        pareto_fig  = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'Cluster {i}' for i in range(5)] + [''],
            specs=[[{'secondary_y': True}] * 3, [{'secondary_y': True}] * 3],
            vertical_spacing=0.18, horizontal_spacing=0.1,
        )
        pr_rows = [1, 1, 1, 2, 2]
        pr_cols = [1, 2, 3, 1, 2]

        for i in range(5):
            dados   = df_m[df_m['Cluster'] == i][cols_pareto].mean()
            s       = dados.abs().sort_values(ascending=False)
            pct     = s / s.sum() * 100
            cumul   = pct.cumsum()
            lsorted = [lbls_pareto[cols_pareto.index(c)] for c in s.index]

            pareto_fig.add_trace(go.Bar(
                x=lsorted, y=pct.values.tolist(),
                marker_color=cl_palette[i], name=f'C{i}', showlegend=False,
                hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
            ), row=pr_rows[i], col=pr_cols[i], secondary_y=False)
            pareto_fig.add_trace(go.Scatter(
                x=lsorted, y=cumul.values.tolist(),
                mode='lines+markers', showlegend=False,
                line=dict(color=ERROR, width=2), marker=dict(size=5, color=ERROR),
                hovertemplate='Cumul.: %{y:.1f}%<extra></extra>',
            ), row=pr_rows[i], col=pr_cols[i], secondary_y=True)

        pareto_fig.update_layout(
            title='Gráfico 29 — Princípio de Pareto por Cluster (K-Means++)',
            paper_bgcolor='white', plot_bgcolor='white',
            height=540, font=dict(size=10, family='Segoe UI'), title_font_size=13,
            margin=dict(t=80, b=60, l=50, r=60),
        )
        for _r in [1, 2]:
            for _c in [1, 2, 3]:
                pareto_fig.update_yaxes(range=[0, 110], secondary_y=True, row=_r, col=_c)

        return (corr_fig, pca_var_fig, kdist_fig, elbow_sil_fig,
                silh_diag_fig, pca2d_fig, shap_fig, pareto_fig)

    except Exception as e:
        print(f'[_build_cluster_charts error] {e}')
        import traceback; traceback.print_exc()
        return tuple([_ph] * 8)


(_CL_CORR_FIG, _CL_PCA_VAR_FIG, _CL_KDIST_FIG, _CL_ELBOW_SIL_FIG,
 _CL_SILH_DIAG_FIG, _CL_PCA2D_FIG, _CL_SHAP_FIG, _CL_PARETO_FIG) = _build_cluster_charts()

_GRAPH_CFG = {'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}

# ─── Pré-computação dos gráficos de clusterização ────────────────────────────
CLUSTER_MODEL_PATH = os.path.join(
    ROOT_DIR, '..', 'modelos', 'modelo_clusterizacao', 'modelo_kmeans_personas.pkl'
)

def _build_cluster_charts():
    """
    Gera os 8 gráficos interativos de clusterização no startup.
    Usa sklearn (disponível no venv) + pandas + plotly.
    Retorna: (pca_var_fig, corr_fig, kdist_fig, elbow_fig,
               sil_fig, viz2d_fig, shap_fig, pareto_fig)
    """
    from sklearn.decomposition import PCA as _PCA
    from sklearn.neighbors import NearestNeighbors as _NN
    from sklearn.metrics import silhouette_samples as _sil_samples
    from sklearn.ensemble import RandomForestClassifier as _RF
    from plotly.subplots import make_subplots

    _ph = go.Figure().update_layout(
        paper_bgcolor='white', height=360,
        annotations=[dict(text='Dados de clusterização não disponíveis',
                          x=0.5, y=0.5, xref='paper', yref='paper',
                          showarrow=False, font=dict(size=13, color='#aaa'))],
        margin=dict(t=20, b=20, l=20, r=20),
    )

    try:
        art    = joblib.load(CLUSTER_MODEL_PATH)
        scaler = art['scaler']
        pca_model  = art['pca']      # 6 componentes (85% variância)
        kmeans     = art['modelo']   # K=5
        feat_names = art['feature_names']
        personas   = art['personas']

        CL_COLORS = {0: '#4C9BE8', 1: '#E8834C', 2: '#4CAF50', 3: '#9B59B6', 4: '#E74C3C'}
        CL_NAMES  = {k: v['nome'] for k, v in personas.items()}
        feat_labels = [f.replace('_', ' ') for f in feat_names]

        # Carregar e preparar dados
        df = pd.read_parquet(DATA_PATH)
        city_map = {'Tier_1': 3, 'Tier_2': 2, 'Tier_3': 1}
        df['City_Tier_enc'] = df['City_Tier'].map(city_map)

        df_modelo = pd.DataFrame({
            'Rent_pct':            df['Rent'] / df['Income'],
            'Loan_Repayment_pct':  df['Loan_Repayment'] / df['Income'],
            'Disposable_pct':      df['Disposable_Income'] / df['Income'],
            'Desired_Savings_pct': df['Desired_Savings_Percentage'] / 100,
            'Age':                 df['Age'],
            'Dependents':          df['Dependents'],
            'Gap_Poupanca':        df['Desired_Savings_Percentage'] / 100
                                   - df['Disposable_Income'] / df['Income'],
            'City_Tier_enc':       df['City_Tier_enc'],
            'Gastos_Consumo_pct':  (df['Groceries'] + df['Eating_Out'] + df['Entertainment']) / df['Income'],
            'Gastos_Fixos_pct':    (df['Insurance'] + df['Transport'] + df['Healthcare']) / df['Income'],
        })

        df_norm     = scaler.transform(df_modelo)
        df_pca_full = pca_model.transform(df_norm)   # (20000, 6)
        labels      = kmeans.predict(df_pca_full)     # (20000,)

        _ly = dict(paper_bgcolor='white', plot_bgcolor='white',
                   font=dict(size=12, family='Segoe UI'), title_font_size=13)

        # ── 1. PCA: Variância Acumulada (bloco 1.7) ───────────────────────────
        pca_full = _PCA()
        pca_full.fit(df_norm)
        var_acum = pca_full.explained_variance_ratio_.cumsum() * 100
        n_comps  = list(range(1, len(var_acum) + 1))

        pca_fig = go.Figure()
        pca_fig.add_trace(go.Scatter(
            x=n_comps, y=var_acum.tolist(), mode='lines+markers',
            line=dict(color=ROXO, width=2.5), marker=dict(size=8),
            name='Variância Acumulada',
            hovertemplate='<b>%{x} componentes</b><br>Variância: %{y:.1f}%<extra></extra>',
        ))
        pca_fig.add_hline(y=85, line_dash='dash', line_color='red',
                          annotation_text='85%', annotation_position='top right')
        pca_fig.add_hline(y=95, line_dash='dash', line_color='green',
                          annotation_text='95%', annotation_position='top right')
        pca_fig.update_layout(
            **_ly, title='Gráfico 1.7 — Variância Acumulada pelo PCA',
            xaxis_title='Número de Componentes', yaxis=dict(title='Variância Acumulada (%)', range=[0, 106]),
            height=400, margin=dict(t=60, b=50, l=80, r=60),
        )
        pca_fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0', dtick=1)
        pca_fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

        # ── 2. Correlação das Features Transformadas (bloco 1.6) ──────────────
        corr_cl   = df_modelo.corr().round(2)
        cl_labels = [f.replace('_', ' ') for f in df_modelo.columns]

        corr_cl_fig = px.imshow(
            corr_cl.values, x=cl_labels, y=cl_labels,
            text_auto='.2f', color_continuous_scale='RdYlGn', range_color=[-1, 1],
            title='Gráfico 1.6 — Correlação das Features Transformadas (Clusterização)',
        )
        corr_cl_fig.update_layout(
            **_ly, height=520, margin=dict(t=60, b=80, l=170, r=40),
            coloraxis_colorbar=dict(title='r'),
        )
        corr_cl_fig.update_xaxes(tickangle=-40)
        corr_cl_fig.update_traces(
            hovertemplate='<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:.2f}<extra></extra>'
        )

        # ── 3. K-Distance — Epsilon DBSCAN (bloco 2.1) ───────────────────────
        nn   = _NN(n_neighbors=12)
        nn.fit(df_pca_full)
        dist, _ = nn.kneighbors(df_pca_full)
        dist_sorted = np.sort(dist[:, 11])

        kdist_fig = go.Figure()
        kdist_fig.add_trace(go.Scatter(
            x=list(range(len(dist_sorted))), y=dist_sorted.tolist(),
            mode='lines', line=dict(color=ROXO, width=1.5),
            name='Distância ao 12º vizinho',
            hovertemplate='Ponto %{x}<br>Distância: %{y:.3f}<extra></extra>',
        ))
        kdist_fig.add_hline(y=0.70, line_dash='dash', line_color=ERROR,
                            annotation_text='ε = 0.70 (selecionado)',
                            annotation_position='top right')
        kdist_fig.update_layout(
            **_ly, title='Gráfico 2.1 — K-Distance: Descobrindo o Epsilon Ideal',
            xaxis_title='Pontos ordenados',
            yaxis_title='Distância ao 12º vizinho mais próximo',
            height=400, margin=dict(t=60, b=50, l=80, r=60),
        )
        kdist_fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        kdist_fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

        # ── 4. Elbow + Silhouette Score (bloco 3.1) — hardcoded do notebook ───
        K_range     = list(range(2, 11))
        inertias    = [64141, 56470, 51190, 46746, 42896, 40692, 39120, 37506, 36251]
        silhouettes = [0.2290, 0.2105, 0.2214, 0.2163, 0.1847, 0.1733, 0.1701, 0.1600, 0.1590]

        elbow_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Método do Cotovelo (Elbow)', 'Silhouette Score'),
        )
        elbow_fig.add_trace(go.Scatter(
            x=K_range, y=inertias, mode='lines+markers',
            line=dict(color='steelblue', width=2.5), marker=dict(size=8),
            name='Inércia',
            hovertemplate='K=%{x}<br>Inércia: %{y:,.0f}<extra></extra>',
        ), row=1, col=1)
        elbow_fig.add_trace(go.Scatter(
            x=K_range, y=silhouettes, mode='lines+markers',
            line=dict(color=ERROR, width=2.5),
            marker=dict(size=8, symbol='square', color=ERROR),
            name='Silhouette',
            hovertemplate='K=%{x}<br>Silhouette: %{y:.4f}<extra></extra>',
        ), row=1, col=2)
        elbow_fig.add_vline(x=5, line_dash='dash', line_color='gray',
                            annotation_text='K=5', annotation_position='top right')
        elbow_fig.update_layout(
            **_ly, title='Gráfico 3.1 — Método do Cotovelo e Silhouette Score',
            height=420, margin=dict(t=70, b=60, l=70, r=40),
            legend=dict(orientation='h', y=-0.2),
        )
        elbow_fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0', dtick=1,
                               title_text='Número de Clusters (K)')
        elbow_fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        elbow_fig.update_yaxes(title_text='Inércia', row=1, col=1)
        elbow_fig.update_yaxes(title_text='Score', row=1, col=2)

        # ── 5. Silhouette Diagram K=3,4,5 (bloco 3.2) ───────────────────────
        from matplotlib import cm as _mpl_cm
        import matplotlib.colors as _mcolors
        from sklearn.cluster import KMeans as _KMeans

        def _nipy_hex(i, k):
            return _mcolors.to_hex(_mpl_cm.nipy_spectral(float(i) / k))

        sil_fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['K = 3', 'K = 4', 'K = 5'],
        )
        _xrefs = ['x', 'x2', 'x3']
        _yrefs = ['y', 'y2', 'y3']

        for col_idx, k in enumerate([3, 4, 5], start=1):
            km_k    = _KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            lbl_k   = km_k.fit_predict(df_pca_full)
            sv_k    = _sil_samples(df_pca_full, lbl_k)
            avg_k   = float(sv_k.mean())
            y_lower = 10

            for i in range(k):
                cl_vals = np.sort(sv_k[lbl_k == i])
                n_cl    = len(cl_vals)
                y_upper = y_lower + n_cl
                col_hex = _nipy_hex(i, k)
                y_pts   = np.linspace(y_lower, y_upper, n_cl)

                x_fill = np.concatenate([[0], cl_vals, [0, 0]])
                y_fill = np.concatenate([[y_lower], y_pts, [y_upper, y_lower]])

                sil_fig.add_trace(go.Scatter(
                    x=x_fill.tolist(), y=y_fill.tolist(),
                    fill='toself', fillcolor=col_hex, opacity=0.7,
                    line=dict(color=col_hex, width=0.5),
                    name=f'K={k} Cl{i}', showlegend=False,
                    hovertemplate=(f'K={k} Cluster {i}<br>'
                                   'Silhouette: %{x:.3f}<extra></extra>'),
                ), row=1, col=col_idx)

                sil_fig.add_annotation(
                    x=-0.07, y=y_lower + 0.5 * n_cl,
                    xref=_xrefs[col_idx - 1], yref=_yrefs[col_idx - 1],
                    text=str(i), showarrow=False,
                    font=dict(size=11, color=col_hex, family='Segoe UI'),
                )
                y_lower = y_upper + 10

            sil_fig.add_vline(
                x=avg_k, line_dash='dash', line_color='red', col=col_idx, row=1,
                annotation_text=f'Média: {avg_k:.3f}',
                annotation_position='top right',
            )

        sil_fig.update_layout(
            **_ly,
            title='Gráfico 3.2 — Silhouette Diagram — Comparação de K',
            height=500, margin=dict(t=80, b=50, l=60, r=40),
        )
        sil_fig.update_xaxes(
            range=[-0.15, 1], showgrid=True, gridcolor='#f0f0f0',
            title_text='Silhouette coefficient',
        )
        sil_fig.update_yaxes(showticklabels=False, showgrid=False)

        # ── 6. Visualização 2D dos dados (bloco 1.8) ─────────────────────────
        pca2d  = _PCA(n_components=2, random_state=42)
        df_2d  = pca2d.fit_transform(df_norm)

        viz2d_fig = go.Figure(go.Scatter(
            x=df_2d[:, 0].tolist(), y=df_2d[:, 1].tolist(),
            mode='markers',
            marker=dict(color='steelblue', size=3, opacity=0.3),
            hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
        ))
        viz2d_fig.update_layout(
            **_ly,
            title='Visualização 2D dos dados (PCA)',
            xaxis_title='Componente Principal 1',
            yaxis_title='Componente Principal 2',
            height=500, margin=dict(t=60, b=60, l=70, r=40),
        )
        viz2d_fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        viz2d_fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

        # ── 7. SHAP Beeswarm Global (bloco 3.10.1) ────────────────────────────
        import shap as _shap

        rng_s      = np.random.RandomState(42)
        sample_idx = rng_s.choice(len(df_modelo), 2000, replace=False)
        X_smp      = df_modelo.iloc[sample_idx]
        lbl_smp    = labels[sample_idx]

        rf = _RF(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        rf.fit(X_smp, lbl_smp)

        explainer   = _shap.TreeExplainer(rf)
        shap_raw    = explainer.shap_values(X_smp)   # (2000, 10, 5)
        shap_arr    = np.array(shap_raw)
        shap_mean   = shap_arr.mean(axis=2)           # (2000, 10) média sobre classes

        mean_abs    = np.abs(shap_mean).mean(axis=0)  # (10,) importância por feature
        feat_order  = np.argsort(mean_abs)            # crescente → fundo-para-cima

        rng_b = np.random.RandomState(0)
        all_x, all_y, all_color = [], [], []
        for rank, fi in enumerate(feat_order):
            x_vals    = shap_mean[:, fi]
            fv        = X_smp.iloc[:, fi].values
            fv_range  = fv.max() - fv.min()
            fv_norm   = (fv - fv.min()) / fv_range if fv_range > 0 else np.full_like(fv, 0.5)
            jitter    = rng_b.uniform(-0.38, 0.38, len(x_vals))
            all_x.extend(x_vals.tolist())
            all_y.extend((rank + jitter).tolist())
            all_color.extend(fv_norm.tolist())

        shap_cl_fig = go.Figure(go.Scatter(
            x=all_x, y=all_y,
            mode='markers',
            marker=dict(
                size=3, opacity=0.55,
                color=all_color,
                colorscale=[[0, '#3182bd'], [0.5, '#bdbdbd'], [1, '#e6550d']],
                cmin=0, cmax=1,
                colorbar=dict(
                    title='Feature<br>value',
                    tickvals=[0.05, 0.95], ticktext=['Low', 'High'],
                    len=0.5, y=0.5, thickness=14,
                ),
            ),
            hovertemplate='SHAP: %{x:.6f}<extra></extra>',
        ))
        shap_cl_fig.update_layout(
            **_ly,
            title=('SHAP Beeswarm Global — Importância das Variáveis<br>'
                   '<sup>(Surrogate RandomForest sobre K-Means++)</sup>'),
            xaxis_title='SHAP value (impact on model output)',
            yaxis=dict(
                tickvals=list(range(len(feat_names))),
                ticktext=[feat_names[fi].replace('_', ' ') for fi in feat_order],
                showgrid=False,
            ),
            height=480, margin=dict(t=80, b=60, l=160, r=80),
            showlegend=False,
        )
        shap_cl_fig.update_xaxes(
            showgrid=True, gridcolor='#f0f0f0',
            zeroline=True, zerolinecolor='#cccccc', zerolinewidth=1,
        )

        # ── 8. Pareto por Cluster com eixo Y duplo (bloco 3.8) ────────────────
        colunas_pareto = ['Rent_pct', 'Loan_Repayment_pct',
                          'Gastos_Consumo_pct', 'Gastos_Fixos_pct']
        cores_pareto   = ['steelblue', 'coral', 'green', 'purple', 'orange']
        df_m_tmp       = df_modelo.copy()
        df_m_tmp['_Cluster'] = labels

        specs_p = [[{"secondary_y": True}] * 3,
                   [{"secondary_y": True}, {"secondary_y": True}, {}]]
        pareto_fig = make_subplots(
            rows=2, cols=3,
            specs=specs_p,
            subplot_titles=[f'Pareto — Cluster {i}' for i in range(5)] + [''],
        )
        row_col = [(1,1),(1,2),(1,3),(2,1),(2,2)]

        for i, (r, c) in enumerate(row_col):
            dados      = df_m_tmp[df_m_tmp['_Cluster'] == i][colunas_pareto].mean()
            dados_sort = dados.abs().sort_values(ascending=False)
            pct        = (dados_sort / dados_sort.sum() * 100).values
            cum        = np.cumsum(pct)
            lbls       = [x for x in dados_sort.index.tolist()]

            pareto_fig.add_trace(go.Bar(
                x=lbls, y=pct.tolist(),
                marker_color=cores_pareto[i], opacity=0.7,
                name=f'Cluster {i}', showlegend=False,
                hovertemplate='<b>%{x}</b><br>%{y:.1f}% do total<extra></extra>',
            ), row=r, col=c, secondary_y=False)

            pareto_fig.add_trace(go.Scatter(
                x=lbls, y=cum.tolist(),
                mode='lines+markers',
                line=dict(color='black', width=2),
                marker=dict(size=6, color='black'),
                name='Cumulativo', showlegend=(i == 0),
                hovertemplate='<b>%{x}</b><br>Cumulativo: %{y:.1f}%<extra></extra>',
            ), row=r, col=c, secondary_y=True)

            pareto_fig.add_trace(go.Scatter(
                x=lbls, y=[80] * len(lbls),
                mode='lines', line=dict(color='red', dash='dash', width=1.5),
                name='80%', showlegend=(i == 0), hoverinfo='skip',
            ), row=r, col=c, secondary_y=True)

            pareto_fig.update_yaxes(
                title_text='% do total', row=r, col=c, secondary_y=False,
            )
            pareto_fig.update_yaxes(
                range=[0, 110], title_text='% acumulado',
                row=r, col=c, secondary_y=True,
            )

        pareto_fig.update_layout(
            **_ly,
            title='Gráfico 3.8 — Princípio de Pareto: Comprometimento Financeiro por Persona',
            height=820, margin=dict(t=80, b=100, l=60, r=60),
            legend=dict(orientation='h', y=-0.06),
        )
        pareto_fig.update_xaxes(tickangle=-45)

        return (pca_fig, corr_cl_fig, kdist_fig, elbow_fig,
                sil_fig, viz2d_fig, shap_cl_fig, pareto_fig)

    except Exception:
        import traceback; traceback.print_exc()
        return (_ph,) * 8


(_CLUST_PCA_FIG, _CLUST_CORR_FIG, _CLUST_KDIST_FIG, _CLUST_ELBOW_FIG,
 _CLUST_SIL_FIG,  _CLUST_2D_FIG,   _CLUST_SHAP_FIG,  _CLUST_PARETO_FIG
 ) = _build_cluster_charts()


def kdd_tag(step):
    colors = {
        'Processamento': '#6366f1',
        'Transformação':  '#0891b2',
        'Mineração':      '#7c3aed',
        'Resultados':     '#059669',
    }
    return html.Span(f'KDD · {step}', style={
        'backgroundColor': colors.get(step, ROXO),
        'color': BRANCO, 'borderRadius': '4px',
        'padding': '3px 10px', 'fontSize': '11px',
        'fontWeight': '700', 'letterSpacing': '0.5px',
        'display': 'inline-block', 'marginBottom': '16px',
    })

# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICAÇÃO,seções KDD
# ══════════════════════════════════════════════════════════════════════════════

def classif_processamento():
    cond_headers = ['Condição', 'Descrição', 'Fórmula', 'Limiar']
    cond_rows = [
        ['C1', 'Endividamento Elevado',               'perc_emprestimo  =  Loan_Repayment / Income',                           '> 10%'],
        ['C2', 'Buffer de Segurança Baixo',           'perc_buffer  =  (Disposable_Income − Desired_Savings) / Income',        '< 10%'],
        ['C3', 'Gastos Não Essenciais Altos',         'perc_nao_essenciais  =  (Eating_Out + Entertainment) / Income',         '> 8,5%'],
        ['C4', 'Potencial de Economia em Groceries',  'perc_pot_groceries  =  Potential_Savings_Groceries / Income',           '> 8%'],
    ]

    return html.Div([
        kdd_tag('Processamento'),
        html.H2('Processamento dos Dados', style=H2),
        html.P(
            'Como o dataset não possuía uma variável-alvo pronta, a variável Vulnerable foi construída '
            'a partir de regras de negócio validadas empiricamente. Trata-se de uma variável binária '
            '(0 = seguro, 1 = vulnerável): o cliente é classificado como vulnerável quando satisfaz '
            'duas ou mais das quatro condições de risco abaixo. O objetivo do modelo é identificar '
            'os clientes seguros (classe 0), bons candidatos para concessão de crédito.',
            style=P_STYLE
        ),

        html.Div([
            html.H3('Indicadores Auxiliares para Construção do Target', style=H3),
            html.P(
                'Quatro indicadores derivados da renda foram criados exclusivamente para definir a '
                'variável-alvo. Eles não entram no modelo de predição (seriam leakage), mas fundamentam '
                'a lógica de negócio que classifica cada cliente:',
                style={**P_STYLE, 'marginBottom': '16px'}
            ),
            html.Div([
                html.Div([
                    html.Div('perc_emprestimo', style={
                        'fontFamily': 'monospace', 'fontSize': '13px', 'fontWeight': '700',
                        'color': ROXO, 'marginBottom': '8px',
                        'backgroundColor': 'rgba(29,18,82,0.06)', 'display': 'inline-block',
                        'padding': '3px 10px', 'borderRadius': '4px',
                    }),
                    html.Div([html.Code('Loan_Repayment / Income', style={'fontSize': '12px', 'color': '#7c3aed'})],
                             style={'marginBottom': '6px'}),
                    html.Div('Percentual da renda comprometida com empréstimos. '
                             'Acima de 10% ativa a condição C1 de risco.',
                             style={'color': TEXT_MUTED, 'fontSize': '13px', 'lineHeight': '1.6'}),
                ], style={'flex': '1', 'padding': '18px', 'backgroundColor': BG_PAGE,
                          'borderRadius': '8px', 'borderLeft': f'3px solid {ROXO}'}),

                html.Div([
                    html.Div('perc_buffer', style={
                        'fontFamily': 'monospace', 'fontSize': '13px', 'fontWeight': '700',
                        'color': '#0891b2', 'marginBottom': '8px',
                        'backgroundColor': 'rgba(8,145,178,0.06)', 'display': 'inline-block',
                        'padding': '3px 10px', 'borderRadius': '4px',
                    }),
                    html.Div([html.Code('(Disposable_Income − Desired_Savings) / Income', style={'fontSize': '12px', 'color': '#7c3aed'})],
                             style={'marginBottom': '6px'}),
                    html.Div('Margem livre após descontar a meta de poupança. '
                             'Abaixo de 10% ativa a condição C2 de risco.',
                             style={'color': TEXT_MUTED, 'fontSize': '13px', 'lineHeight': '1.6'}),
                ], style={'flex': '1', 'padding': '18px', 'backgroundColor': BG_PAGE,
                          'borderRadius': '8px', 'borderLeft': '3px solid #0891b2'}),

                html.Div([
                    html.Div('perc_nao_essenciais', style={
                        'fontFamily': 'monospace', 'fontSize': '13px', 'fontWeight': '700',
                        'color': ERROR, 'marginBottom': '8px',
                        'backgroundColor': 'rgba(229,75,75,0.06)', 'display': 'inline-block',
                        'padding': '3px 10px', 'borderRadius': '4px',
                    }),
                    html.Div([html.Code('(Eating_Out + Entertainment) / Income', style={'fontSize': '12px', 'color': '#7c3aed'})],
                             style={'marginBottom': '6px'}),
                    html.Div('Proporção da renda em itens não essenciais. '
                             'Acima de 8,5% ativa a condição C3 de risco.',
                             style={'color': TEXT_MUTED, 'fontSize': '13px', 'lineHeight': '1.6'}),
                ], style={'flex': '1', 'padding': '18px', 'backgroundColor': BG_PAGE,
                          'borderRadius': '8px', 'borderLeft': f'3px solid {ERROR}'}),

                html.Div([
                    html.Div('perc_pot_groceries', style={
                        'fontFamily': 'monospace', 'fontSize': '13px', 'fontWeight': '700',
                        'color': SUCCESS, 'marginBottom': '8px',
                        'backgroundColor': 'rgba(46,125,50,0.06)', 'display': 'inline-block',
                        'padding': '3px 10px', 'borderRadius': '4px',
                    }),
                    html.Div([html.Code('Potential_Savings_Groceries / Income', style={'fontSize': '12px', 'color': '#7c3aed'})],
                             style={'marginBottom': '6px'}),
                    html.Div('Potencial de economia em supermercado em relação à renda. '
                             'Acima de 8% indica C4 de risco.',
                             style={'color': TEXT_MUTED, 'fontSize': '13px', 'lineHeight': '1.6'}),
                ], style={'flex': '1', 'padding': '18px', 'backgroundColor': BG_PAGE,
                          'borderRadius': '8px', 'borderLeft': f'3px solid {SUCCESS}'}),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(240px, 1fr))',
                      'gap': '16px'}),
        ], style=CARD),

        html.Div([
            html.H3('Regras de Negócio para Construção do Target', style=H3),
            html.P('O Risk Score de cada cliente varia de 0 a 4. Se o somatório das condições ativas for ≥ 2, '
                   'a variável Vulnerable recebe valor 1 (alto risco). Caso contrário, valor 0 (seguro). '
                   'O modelo é treinado para identificar os clientes seguros (0), perfil favorável para crédito.',
                   style={**P_STYLE, 'marginBottom': '20px'}),
            html_table(cond_headers, cond_rows),
            info_box([
                html.Strong('Critério de ativação: '),
                'Risk Score ≥ 2  →  Vulnerable = 1 (evitar)  |  Risk Score < 2  →  Vulnerable = 0 (seguro, alvo do modelo)'
            ], border_color=SUCCESS),
        ], style=CARD),

        html.Div([
            html.H3('Distribuição da Base após Criação do Target', style=H3),
            html.P('Após remover os 112 clientes com Desired_Savings = 0 (sem meta de poupança definida), a base de análise passa a ter 19.888 registros.', style={**P_STYLE, 'marginBottom': '16px'}),
            html.Div([
                metric_card('Registros analisados', '19.888', ROXO),
                metric_card('Seguros (classe 0)', '15.790  (~79,4%)', SUCCESS),
                metric_card('Vulneráveis (classe 1)', '4.098  (~20,6%)', ERROR),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '16px'}),
        ], style=CARD),
    ])


def classif_transformacao():
    return html.Div([
        kdd_tag('Transformação'),
        html.H2('Transformação dos Dados', style=H2),

        html.Div([
            html.H3('Problema: Target Leakage', style={**H3, 'color': ERROR}),
            html.P(
                'A variável-alvo (Vulnerable) foi construída a partir de quatro indicadores derivados '
                '(perc_emprestimo, perc_buffer, perc_nao_essenciais, perc_pot_groceries). Incluí-los '
                'diretamente no modelo permitiria a engenharia reversa da regra de negócio, os '
                'algoritmos memorizariam o resultado em vez de aprender padrões preditivos genuínos.',
                style=P_STYLE
            ),
            info_box([
                html.Strong('Decisão: '), 'Os quatro indicadores usados no target foram excluídos. '
                'A matriz final usa ', html.Strong('9 variáveis percentuais independentes'),
                ' (rácios de gastos reais sobre a renda), sem nenhum dos componentes da fórmula do target.'
            ], border_color=ERROR),
        ], style=CARD),

        html.Div([
            html.H3('Engenharia de Features, Conversão para Rácios', style=H3),
            html.P(
                'Cada gasto foi dividido pela renda do cliente, transformando valores absolutos em '
                'proporções comparáveis. Isso neutraliza o viés de escala: um aluguel de R$2.000 '
                'representa comprometimentos radicalmente distintos para quem ganha R$4.000 ou R$30.000. '
                'Com rácios, o modelo aprende padrões comportamentais, não diferenças de poder aquisitivo.',
                style=P_STYLE
            ),
            html.Div([
                html.Span(f, style={
                    'backgroundColor': 'rgba(46,125,50,0.08)', 'color': SUCCESS,
                    'borderRadius': '6px', 'padding': '7px 16px', 'fontSize': '13px', 'fontWeight': '600',
                    'display': 'inline-block',
                }) for f in FEATURES
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '12px 14px', 'marginTop': '16px'}),
        ], style=CARD),

        html.Div([
            html.H3('Análise de Multicolinearidade entre Features', style=H3),
            html.P(
                'A matriz de correlação das 9 features preditoras confirma ausência de multicolinearidade '
                'severa (nenhum par com |r| > 0,7), validando que os rácios são numericamente '
                'independentes entre si. Isso torna o modelo de Regressão Logística matematicamente '
                'estável sem necessidade de regularização adicional.',
                style=P_STYLE
            ),
            dcc.Graph(id='ml-corr-fig', figure=_CORR_FIG, config=_GRAPH_CFG,
                      style={'marginTop': '16px'}),
        ], style=CARD),
    ])


def classif_mineracao():
    alg_rows = [
        ['Regressão Logística ★', 'Linear',   'Interpretável, maior AUC-ROC, selecionado'],
        ['SVM',                   'Kernel',   'Pipeline com StandardScaler interno'],
        ['Random Forest',         'Ensemble', 'Bagging de árvores de decisão'],
        ['Gradient Boosting',     'Ensemble', 'Boosting sequencial com subsampling'],
        ['XGBoost',               'Ensemble', 'Boosting otimizado com regularização'],
        ['CatBoost',              'Ensemble', 'Boosting com tratamento nativo de categorias'],
    ]

    return html.Div([
        kdd_tag('Mineração'),
        html.H2('Mineração de Dados e Modelagem', style=H2),

        html.Div([
            html.H3('Divisão Estratificada da Base', style=H3),
            html.P(
                'A base foi dividida de forma estratificada (random_state=42), preservando a proporção '
                'original entre as classes em treino e teste:',
                style=P_STYLE
            ),
            html.Div([
                metric_card('Treino (80%)', '15.910 registros', ROXO),
                metric_card('Teste  (20%)', '3.978 registros',  '#6366f1'),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px'}),
        ], style=CARD),

        html.Div([
            html.H3('Balanceamento de Classes', style=H3),
            html.P(
                'A base é desbalanceada (~79% seguros, ~21% vulneráveis). Para evitar que os modelos '
                'ignorem a classe minoritária, foi aplicado um peso de classe proporcional ao inverso '
                'da frequência: class_weight = {0: 1, 1: ratio × 0.8}, onde ratio = n_seguros / n_vulneráveis. '
                'O SVM recebe esse mesmo peso via pipeline com StandardScaler interno.',
                style=P_STYLE
            ),
        ], style=CARD),

        html.Div([
            html.H3('Algoritmos Avaliados, 6 Famílias', style=H3),
            html.P(
                'Foram testadas seis famílias de aprendizado de máquina para garantir diversidade '
                'analítica e identificar a arquitetura que melhor generaliza o padrão de crédito seguro:',
                style=P_STYLE
            ),
            html_table(['Algoritmo', 'Paradigma', 'Justificativa'], alg_rows),
        ], style=CARD),

        html.Div([
            html.H3('Critério de Seleção, Recall e Precision para Seguro', style={**H3, 'color': SUCCESS}),
            html.P(
                'As métricas prioritárias do critério de seleção foram Recall e Precisão para a classe Seguro (0): '
                'recall mede quantos clientes seguros o modelo efetivamente identifica, enquanto precisão mede '
                'quantos dos aprovados são de fato seguros. O F1-Score sintetiza o equilíbrio entre ambas e o '
                'AUC-ROC avalia a separabilidade geral. A Regressão Logística obteve o maior AUC-ROC (0,6529) '
                'entre todos os modelos testados, com Recall = 0,80 e Precisão = 0,83 para a classe Seguro, '
                'garantindo que bons clientes sejam aprovados sem expor a carteira a risco.',
                style=P_STYLE
            ),
            info_box([
                html.Strong('Lógica de negócio: '),
                'identificar corretamente os clientes seguros permite expandir a concessão de crédito com '
                'confiança. Falsos negativos (seguros classificados como vulneráveis) representam oportunidades '
                'perdidas; falsos positivos (vulneráveis liberados) representam risco de inadimplência. '
                'O modelo busca o equilíbrio ótimo entre esses dois erros.'
            ], border_color=SUCCESS),
        ], style=CARD),
    ])


def classif_resultados():
    # Tabela consolidada — 6 algoritmos, métricas reais do notebook
    metrics_headers = ['Modelo', 'Métrica', 'Seguro (0)', 'Vulnerável (1)', 'Média', 'Ponderada']
    metrics_rows = [
        ['Regressão Logística ★', 'Precisão',  '0,830', '0,310', '0,570', '0,720'],
        ['Regressão Logística ★', 'Recall',    '0,800', '0,350', '0,570', '0,710'],
        ['Regressão Logística ★', 'F1-Score',  '0,810', '0,330', '0,570', '0,710'],
        ['Regressão Logística ★', 'Acurácia',  '—',     '—',     '0,710', '—'],
        ['Regressão Logística ★', 'AUC-ROC',   '—',     '—',     '—',     '0,6529'],
        ['Random Forest',         'Precisão',  '0,800', '0,530', '0,660', '0,740'],
        ['Random Forest',         'Recall',    '1,000', '0,010', '0,500', '0,790'],
        ['Random Forest',         'F1-Score',  '0,880', '0,020', '0,450', '0,710'],
        ['Random Forest',         'Acurácia',  '—',     '—',     '0,790', '—'],
        ['Random Forest',         'AUC-ROC',   '—',     '—',     '—',     '0,6144'],
        ['Gradient Boosting',     'Precisão',  '0,790', '0,330', '0,560', '0,700'],
        ['Gradient Boosting',     'Recall',    '1,000', '0,000', '0,500', '0,790'],
        ['Gradient Boosting',     'F1-Score',  '0,880', '0,000', '0,440', '0,700'],
        ['Gradient Boosting',     'Acurácia',  '—',     '—',     '0,790', '—'],
        ['Gradient Boosting',     'AUC-ROC',   '—',     '—',     '—',     '0,6512'],
        ['XGBoost',               'Precisão',  '0,820', '0,280', '0,550', '0,710'],
        ['XGBoost',               'Recall',    '0,760', '0,360', '0,560', '0,680'],
        ['XGBoost',               'F1-Score',  '0,790', '0,320', '0,550', '0,690'],
        ['XGBoost',               'Acurácia',  '—',     '—',     '0,680', '—'],
        ['XGBoost',               'AUC-ROC',   '—',     '—',     '—',     '0,6099'],
        ['SVM',                   'Precisão',  '0,840', '0,300', '0,570', '0,730'],
        ['SVM',                   'Recall',    '0,710', '0,480', '0,590', '0,660'],
        ['SVM',                   'F1-Score',  '0,770', '0,370', '0,570', '0,690'],
        ['SVM',                   'Acurácia',  '—',     '—',     '0,660', '—'],
        ['SVM',                   'AUC-ROC',   '—',     '—',     '—',     '0,6431'],
        ['CatBoost',              'Precisão',  '0,820', '0,290', '0,560', '0,710'],
        ['CatBoost',              'Recall',    '0,760', '0,380', '0,570', '0,680'],
        ['CatBoost',              'F1-Score',  '0,790', '0,330', '0,560', '0,690'],
        ['CatBoost',              'Acurácia',  '—',     '—',     '0,680', '—'],
        ['CatBoost',              'AUC-ROC',   '—',     '—',     '—',     '0,6267'],
    ]

    # Comparativo final — ordenado por F1 (Seguro) decrescente
    comp_headers = ['Modelo', 'Precisão (Seguro)', 'Recall (Seguro)', 'F1 (Seguro)']
    comp_rows = [
        ['Random Forest',         '0,800', '1,000', '0,880'],
        ['Gradient Boosting',     '0,790', '1,000', '0,880'],
        ['Regressão Logística ★', '0,830', '0,800', '0,810'],
        ['XGBoost',               '0,820', '0,760', '0,790'],
        ['CatBoost',              '0,820', '0,760', '0,790'],
        ['SVM',                   '0,840', '0,710', '0,770'],
    ]

    # Segmentação por confiança — dados reais do notebook (bloco 13)
    seg_headers = ['Perfil de Confiança', 'Clientes', '% da Base', 'Prob. Média (Seguro)', 'Renda Média', 'Empréstimo Médio', 'Exposição Mensal']
    seg_rows = [
        ['Alta confiança (≥ 70%)',      '2.580',  '13,0%', '73,7%', 'R$ 41.810', 'R$ 2.052', 'R$ 5.294.869'],
        ['Moderada confiança (40–70%)', '15.094', '75,9%', '56,9%', 'R$ 41.439', 'R$ 2.054', 'R$ 31.003.076'],
        ['Baixa confiança (< 40%)',     '2.214',  '11,1%', '37,5%', 'R$ 42.119', 'R$ 1.696', 'R$ 3.754.944'],
    ]

    return html.Div([
        kdd_tag('Resultados'),
        html.H2('Resultados e Análise do Modelo de Classificação', style=H2),

        # Modelo selecionado
        html.Div([
            html.Div([
                html.Div([badge('MODELO SELECIONADO', SUCCESS)], style={'marginBottom': '8px'}),
                html.H3('Regressão Logística, Foco: Identificar Clientes Seguros', style={**H3, 'marginTop': '4px', 'fontSize': '20px'}),
                html.P(
                    'Entre os seis algoritmos comparados, a Regressão Logística foi selecionada por '
                    'apresentar o maior AUC-ROC (0,6529) e o melhor equilíbrio entre precisão e recall '
                    'para a classe Seguro (0). O foco do modelo mudou: em vez de detectar vulneráveis, '
                    'o objetivo agora é identificar bons clientes para concessão de crédito com confiança.',
                    style=P_STYLE
                ),
                html.Div([
                    metric_card('Precisão (Seguro)',  '0,830',  SUCCESS),
                    metric_card('Recall (Seguro)',    '0,800',  SUCCESS),
                    metric_card('F1 (Seguro)',        '0,810',  ROXO),
                    metric_card('AUC-ROC',            '0,6529', '#6366f1'),
                    metric_card('Acurácia Geral',     '71,0%',  '#0891b2'),
                ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))',
                          'gap': '16px', 'marginTop': '16px'}),
            ]),
        ], style=CARD),

        # Tabela consolidada — todos os modelos
        html.Div([
            html.H3('Tabela Consolidada de Métricas, 6 Algoritmos (Tabela 5)', style=H3),
            html.P('Métricas reais obtidas no conjunto de teste (20% da base). ★ indica o modelo selecionado.', style={**P_STYLE, 'marginBottom': '16px'}),
            html.Div(html_table(metrics_headers, metrics_rows),
                     style={'overflowX': 'auto'}),
        ], style=CARD),

        # Comparativo final ordenado por F1 (Seguro)
        html.Div([
            html.H3('Comparativo Final, Ordenado por F1 (Seguro) (Tabela 5b)', style=H3),
            html.P(
                'Ranking dos modelos pela capacidade de identificar clientes seguros, métrica central '
                'para decisão de concessão de crédito. Modelos com recall=1,000 para Seguro (Random Forest '
                'e Gradient Boosting) aprovam praticamente todos os seguros, mas sacrificam a precisão, '
                'aceitam mais vulneráveis por engano. A Regressão Logística oferece o melhor equilíbrio.',
                style={**P_STYLE, 'marginBottom': '16px'}),
            html.Div(html_table(comp_headers, comp_rows, highlight_row=2),
                     style={'overflowX': 'auto'}),
        ], style=CARD),

        # Matriz de Confusão
        html.Div([
            html.H3('Interpretabilidade, Matriz de Confusão', style=H3),
            html.P(
                'O modelo identificou corretamente 2.522 dos 3.158 clientes seguros no conjunto de teste '
                '(Recall Seguro: 79,9%). Os 636 falsos positivos (seguros previstos como vulneráveis) '
                'representam oportunidades de crédito conservadoramente negadas. Os 532 falsos negativos '
                '(vulneráveis aprovados como seguros) representam o risco residual da carteira.',
                style=P_STYLE
            ),
            dcc.Graph(id='ml-cm-fig', figure=_CM_FIG, config=_GRAPH_CFG,
                      style={'marginTop': '16px'}),
        ], style=CARD),

        # SHAP e Pareto
        html.Div([
            html.H3('Explicabilidade, SHAP e Princípio de Pareto', style=H3),
            html.P(
                'Os SHAP values quantificam a contribuição de cada feature para a predição da classe '
                'Seguro em cada cliente. Analisando apenas os clientes verdadeiramente seguros, '
                'Rent_Ratio responde por ~49,5% do poder explicativo e, junto com Groceries_Ratio '
                'e Education_Ratio, cobre mais de 82% da importância acumulada (Princípio de Pareto).',
                style=P_STYLE
            ),
            html.Div([
                html.Div([
                    dcc.Graph(id='ml-shap-fig', figure=_SHAP_FIG, config=_GRAPH_CFG),
                ], style={'flex': '1', 'minWidth': '320px'}),
                html.Div([
                    dcc.Graph(id='ml-pareto-fig', figure=_PARETO_FIG, config=_GRAPH_CFG),
                ], style={'flex': '1', 'minWidth': '320px'}),
            ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap'}),
            info_box([
                html.Strong('Variáveis mais impactantes (perspectiva Seguro): '),
                'Rent_Ratio (44,1%), Education_Ratio (28,8%) e Groceries_Ratio (8,9%), '
                'juntas explicam 81,8% do poder preditivo. São rácios diretamente acionáveis: '
                'clientes com comprometimento de aluguel e educação abaixo de certos limiares '
                'têm perfil financeiro mais estável e favorável à concessão de crédito.'
            ], border_color=SUCCESS),
        ], style=CARD),

        # Cross-Validation
        html.Div([
            html.H3('Cross-Validation, Regressão Logística (5 Folds, Foco: Seguro)', style=H3),
            html.P(
                'A validação cruzada estratificada com 5 folds confirma a estabilidade do modelo: '
                'os desvios-padrão são muito baixos, indicando que o desempenho se generaliza '
                'consistentemente para dados não vistos.',
                style=P_STYLE
            ),
            dcc.Graph(id='ml-cv-fig', figure=_CV_FIG, config=_GRAPH_CFG,
                      style={'marginTop': '8px'}),
        ], style=CARD),

        # Segmentação por probabilidade
        html.Div([
            html.H3('Segmentação por Confiança de Ser Seguro (Tabela 6)', style=H3),
            html.P(
                'O modelo foi aplicado à base completa via predict_proba para segmentar clientes '
                'em três perfis de confiança. Clientes com probabilidade ≥ 70% de serem seguros '
                'representam o público prioritário para produtos de crédito, perfil mais estável '
                'e menor risco de inadimplência.',
                style=P_STYLE
            ),
            html_table(seg_headers, seg_rows),
            info_box([
                html.Strong('Conclusão operacional: '),
                '2.580 clientes com alta confiança de ser seguro (13,0% da base) são candidatos '
                'imediatos para aprovação de crédito, com exposição mensal de R$ 5,29 milhões. '
                'O grupo de moderada confiança (75,9%) pode ser abordado com produtos de menor limite '
                'ou análise complementar, ampliando a base elegível de forma controlada.'
            ], border_color=SUCCESS),
        ], style=CARD),

        # ── Previsão Individual ──────────────────────────────────────────────────
        html.Div([
            html.Div([badge('PREVISÃO INDIVIDUAL')], style={'marginBottom': '12px'}),
            html.H3('Simular Perfil de Crédito de um Cliente', style={**H3, 'marginTop': '0'}),
            html.P(
                'Preencha os dados financeiros do cliente e execute o modelo de Regressão Logística '
                'para obter a probabilidade de ser Seguro e o perfil de elegibilidade para crédito.',
                style={**P_STYLE, 'marginBottom': '24px'}
            ),

            # Grid de inputs — exatamente as 9 variáveis do modelo
            html.Div([
                # Coluna 1 — Perfil
                html.Div([
                    html.Div('Perfil', style={
                        'fontWeight': '700', 'color': ROXO, 'fontSize': '12px',
                        'textTransform': 'uppercase', 'letterSpacing': '0.8px', 'marginBottom': '16px',
                    }),
                    _ip_field('income',     'Renda Mensal (R$)', 35000),
                    _ip_field('age',        'Idade',             35),
                    _ip_field('dependents', 'Nº de Dependentes',  2),
                ], style={'flex': '1', 'minWidth': '200px'}),

                # Coluna 2 — Gastos Mensais (7 variáveis do modelo)
                html.Div([
                    html.Div('Gastos Mensais', style={
                        'fontWeight': '700', 'color': '#0891b2', 'fontSize': '12px',
                        'textTransform': 'uppercase', 'letterSpacing': '0.8px', 'marginBottom': '16px',
                    }),
                    _ip_field('rent',        'Aluguel (R$)',      7000),
                    _ip_field('healthcare',  'Saúde (R$)',         800),
                    _ip_field('education',   'Educação (R$)',     2000),
                    _ip_field('groceries',   'Supermercado (R$)', 3000),
                    _ip_field('transport',   'Transporte (R$)',   1500),
                    _ip_field('utilities',   'Utilidades (R$)',    800),
                    _ip_field('insurance',   'Seguro (R$)',        600),
                ], style={'flex': '1', 'minWidth': '200px'}),
            ], style={'display': 'flex', 'gap': '40px', 'flexWrap': 'wrap', 'marginBottom': '24px',
                      'backgroundColor': BG_PAGE, 'padding': '24px', 'borderRadius': '10px',
                      'border': f'1px solid {BORDER}'}),

            html.Button('Prever Perfil de Crédito', id='ip-predict-btn', n_clicks=0, style={
                'backgroundColor': ROXO, 'color': BRANCO, 'border': 'none',
                'padding': '13px 32px', 'borderRadius': '8px',
                'fontWeight': '700', 'cursor': 'pointer', 'fontSize': '14px',
                'boxShadow': '0 4px 14px rgba(29,18,82,0.3)',
                'transition': 'opacity 0.2s ease',
            }),

            html.Div(id='ip-predict-result', style={'marginTop': '24px'}),
        ], style=CARD),

        # ── Demo na Base Completa ────────────────────────────────────────────────
        html.Div([
            html.Div([badge('DEMO INTERATIVA')], style={'marginBottom': '12px'}),
            html.H3('Rodar o Modelo na Base Completa', style={**H3, 'marginTop': '0'}),
            html.P(
                'Execute o modelo de perfil de crédito na base de 19.888 registros e explore a '
                'distribuição de clientes seguros e vulneráveis com threshold ajustável.',
                style={**P_STYLE, 'marginBottom': '8px'}
            ),
            info_box([
                html.Strong('Intervalo real de probabilidades do modelo: '),
                'A Regressão Logística com AUC-ROC 0,6529 produz p(Vulnerável) entre ',
                html.Strong('0,19 e 0,69'),
                ' — nenhum cliente atinge 70% de probabilidade de vulnerabilidade. '
                'Isso é esperado para discriminação moderada. ',
                html.Strong('Threshold padrão 0,50'),
                ': ~4.700 Vulneráveis / ~15.188 Seguros, alinhado com a matriz de confusão do notebook.'
            ], border_color=ROXO),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span('Threshold p(Vulnerável):', style={'fontWeight': '600'}),
                        html.Span(' '),
                        html.Span('0.50', id='ml-threshold-val', style={
                            'fontWeight': '700', 'color': ROXO,
                            'backgroundColor': '#f0eeff', 'borderRadius': '4px',
                            'padding': '2px 8px', 'fontSize': '13px',
                        }),
                        html.Span('  (intervalo útil: 0,19 – 0,69)',
                                  style={'color': TEXT_MUTED, 'fontSize': '11px', 'marginLeft': '8px'}),
                    ], style={'marginBottom': '10px'}),
                    dcc.Slider(
                        id='ml-threshold', min=0.10, max=0.70, step=0.01, value=0.5,
                        marks={
                            0.10: {'label': '0,10 — máx. Vulneráveis', 'style': {'whiteSpace': 'nowrap', 'color': ERROR}},
                            0.50: {'label': '0,50 — padrão',           'style': {'whiteSpace': 'nowrap', 'color': TEXT_MUTED}},
                            0.69: {'label': '0,69 — máx. prob. modelo','style': {'whiteSpace': 'nowrap', 'color': SUCCESS}},
                        },
                        tooltip={'always_visible': False},
                    ),
                ], style={'flex': '1', 'paddingLeft': '40px', 'paddingRight': '40px'}),
                html.Button('Executar Predição', id='ml-run-btn', style={
                    'backgroundColor': ROXO, 'color': BRANCO, 'border': 'none',
                    'padding': '12px 28px', 'borderRadius': '8px',
                    'fontWeight': '700', 'cursor': 'pointer',
                    'boxShadow': '0 4px 12px rgba(29,18,82,0.3)',
                }),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '32px',
                      'backgroundColor': BG_PAGE, 'padding': '20px', 'borderRadius': '8px',
                      'border': f'1px solid {BORDER}', 'marginBottom': '20px'}),

            html.Div(id='ml-demo-cards'),
            html.Div([
                html.Div(dcc.Graph(
                    id='ml-pie',
                    figure=go.Figure().update_layout(
                        paper_bgcolor='white', plot_bgcolor='white', height=360,
                        annotations=[dict(text='Clique em "Executar Predição"', x=0.5, y=0.5,
                                         xref='paper', yref='paper', showarrow=False,
                                         font=dict(size=13, color='#aaa'))],
                        margin=dict(t=10, b=10, l=10, r=10),
                    ),
                    config={'displayModeBar': False},
                ), style={'flex': '1'}),
                html.Div(dcc.Graph(
                    id='ml-hist',
                    figure=go.Figure().update_layout(
                        paper_bgcolor='white', plot_bgcolor='white', height=360,
                        annotations=[dict(text='Clique em "Executar Predição"', x=0.5, y=0.5,
                                         xref='paper', yref='paper', showarrow=False,
                                         font=dict(size=13, color='#aaa'))],
                        margin=dict(t=10, b=10, l=10, r=10),
                    ),
                    config={'displayModeBar': False},
                ), style={'flex': '1'}),
            ], id='ml-charts-row', style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}),
            dcc.Store(id='ml-store'),
        ], style=CARD),

    ])


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTERIZAÇÃO,seções KDD
# ══════════════════════════════════════════════════════════════════════════════

def cluster_processamento():
    return html.Div([
        kdd_tag('Processamento'),
        html.H2('Processamento dos Dados', style=H2),

        html.Div([
            html.H3('Seleção de Variáveis', style=H3),
            html.P(
                'O modelo de clusterização utilizou a mesma base das onze variáveis preditoras '
                'selecionadas para o modelo de classificação. No entanto, como o objetivo aqui é '
                'identificar perfis comportamentais, e não mensurar risco financeiro absoluto,'
                'essas variáveis foram posteriormente convertidas em proporções relativas à renda.',
                style=P_STYLE
            ),
        ], style=CARD),

        html.Div([
            html.H3('Normalização, Por que RobustScaler e não StandardScaler?', style=H3),
            html.P(
                'As 10 variáveis foram normalizadas via RobustScaler (mediana e IQR), preterindo '
                'o StandardScaler pela presença de outliers financeiros no dataset. O RobustScaler '
                'é insensível a valores extremos, preservando a estrutura dos dados para o '
                'aprendizado não supervisionado.',
                style=P_STYLE
            ),
            html.Div([
                html.Div([
                    html.Div('StandardScaler', style={'fontWeight': '700', 'color': ERROR, 'marginBottom': '8px'}),
                    html.Div('Usa média e desvio padrão → sensível a outliers financeiros extremos.', style={'color': TEXT_MUTED, 'fontSize': '14px'}),
                ], style={'flex': '1', 'padding': '16px', 'backgroundColor': 'rgba(229,75,75,0.05)',
                          'borderRadius': '8px', 'borderLeft': f'4px solid {ERROR}'}),
                html.Div([
                    html.Div('RobustScaler ✓', style={'fontWeight': '700', 'color': SUCCESS, 'marginBottom': '8px'}),
                    html.Div('Usa mediana e IQR → robusto a outliers, ideal para dados financeiros.', style={'color': TEXT_MUTED, 'fontSize': '14px'}),
                ], style={'flex': '1', 'padding': '16px', 'backgroundColor': 'rgba(46,125,50,0.05)',
                          'borderRadius': '8px', 'borderLeft': f'4px solid {SUCCESS}'}),
            ], style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'}),
        ], style=CARD),

        html.Div([
            html.H3('Redução de Dimensionalidade, PCA (85% de Variância)', style=H3),
            html.P(
                'Foi aplicado PCA com threshold de 85% de variância acumulada para reduzir '
                'dimensionalidade e ruído residual antes da clusterização. A inclusão do PCA '
                'foi validada empiricamente pelos resultados do Silhouette Score:',
                style=P_STYLE
            ),
            html.Div([
                metric_card('Silhouette SEM PCA', '0,154', ERROR),
                metric_card('Silhouette COM PCA', '0,217', SUCCESS),
                metric_card('Melhoria',           '+40,9%', ROXO),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '16px'}),
            info_box([
                html.Strong('Conclusão: '),
                'A redução de dimensionalidade via PCA melhora significativamente a qualidade '
                'da segmentação, indicando que as componentes principais capturam estrutura '
                'latente mais relevante do que as variáveis originais.'
            ], border_color=SUCCESS),
            dcc.Graph(id='cl-pca-var-fig', figure=_CL_PCA_VAR_FIG, config=_GRAPH_CFG,
                      style={'marginTop': '16px'}),
        ], style=CARD),
    ])


def cluster_transformacao():
    cat_items = [
        ('Proporcionais', 'rgba(99,102,241,0.1)', '#6366f1',
         'Rent_pct, Loan_Repayment_pct, Disposable_pct, Desired_Savings_pct, frações diretas da renda comprometida.'),
        ('Agregadas', 'rgba(8,145,178,0.1)', '#0891b2',
         'Gastos_Consumo_pct, Gastos_Fixos_pct, soma de divisões menores para reduzir dispersão de dados.'),
        ('Derivada', 'rgba(124,58,237,0.1)', '#7c3aed',
         'Gap_Poupanca, lacuna entre poupança desejada e realizada, mede o desalinhamento financeiro.'),
        ('Contexto', 'rgba(5,150,105,0.1)', '#059669',
         'Age, Dependents, City_Tier_enc, fatores demográficos e de infraestrutura urbana.'),
    ]

    return html.Div([
        kdd_tag('Transformação'),
        html.H2('Transformação dos Dados', style=H2),

        html.Div([
            html.H3('Conversão para Proporções Relativas à Renda', style=H3),
            html.P(
                'A conversão das variáveis absolutas em proporções relativas à renda neutraliza o '
                'viés do poder aquisitivo. Um aluguel de R$2.000 representa comprometimentos '
                'completamente distintos para quem ganha R$4.000 ou R$20.000. Sem essa conversão, '
                'o modelo agruparia indivíduos por nível de renda e não por padrão comportamental.',
                style=P_STYLE
            ),
        ], style=CARD),

        html.Div([
            html.H3('10 Atributos Finais, 4 Categorias', style=H3),
            html.P('As variáveis originais foram transformadas e organizadas em quatro categorias funcionais:', style=P_STYLE),
            html.Div([
                html.Div([
                    html.Div(f'Categoria {i+1}: {name}', style={'fontWeight': '700', 'color': color, 'marginBottom': '8px', 'fontSize': '14px'}),
                    html.Div(desc, style={'color': TEXT_MUTED, 'fontSize': '13px', 'lineHeight': '1.6'}),
                ], style={
                    'padding': '16px', 'borderRadius': '8px',
                    'backgroundColor': bg, 'borderLeft': f'3px solid {color}',
                })
                for i, (name, bg, color, desc) in enumerate(cat_items)
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(280px, 1fr))', 'gap': '16px'}),
        ], style=CARD),

        html.Div([
            html.H3('Matriz de Correlação das Features Transformadas', style=H3),
            html.P(
                'A análise de correlação das proporções revela as relações estruturais entre as '
                'variáveis e confirma que a transformação reduziu o efeito da multicolinearidade '
                'presente nas variáveis brutas.',
                style=P_STYLE
            ),
            dcc.Graph(id='cl-corr-fig', figure=_CL_CORR_FIG, config=_GRAPH_CFG,
                      style={'marginTop': '16px'}),
        ], style=CARD),
    ])


def cluster_mineracao():
    eps_headers = ['Epsilon', 'Clusters Reais', 'Ruído (%)']
    eps_rows = [
        ['0,60', '10', '15,27%'],
        ['0,70', '5',  '6,04%'],
        ['0,85', '3',  '1,85%'],
        ['1,00', '2',  '0,53%'],
        ['1,20', '2',  '0,06%'],
    ]

    return html.Div([
        kdd_tag('Mineração'),
        html.H2('Mineração de Dados e Modelagem', style=H2),

        html.Div([
            html.H3('Paradigma: Dois Algoritmos para Validação Cruzada', style=H3),
            html.P(
                'Foram aplicados dois algoritmos de paradigmas completamente distintos para '
                'verificar se a estrutura dos dados comporta segmentações estáveis sob '
                'diferentes premissas geométricas. A convergência dos dois modelos em um mesmo '
                'número de clusters valida a estrutura de 5 personas como característica '
                'intrínseca do dataset.',
                style=P_STYLE
            ),
        ], style=CARD),

        html.Div([
            html.H3('DBSCAN, Clusterização por Densidade', style=H3),
            html.P(
                'O DBSCAN é baseado em densidade, não requer K predefinido e não assume '
                'geometria esférica. Os parâmetros foram determinados sistematicamente:',
                style=P_STYLE
            ),
            html.Ul([
                html.Li([html.Strong('Epsilon: '), 'determinado via gráfico K-Distance (ponto de inflexão da curva de distâncias).'], style={'marginBottom': '6px', 'color': TEXT_MUTED}),
                html.Li([html.Strong('min_samples: '), 'fixado em 12 após análise empírica da densidade local.'], style={'color': TEXT_MUTED}),
            ], style={'paddingLeft': '20px', 'lineHeight': '1.8', 'fontSize': '15px', 'marginBottom': '20px'}),
            html.H4('Teste de Múltiplos Valores de Epsilon (Tabela 7)', style={'color': TEXT_MAIN, 'marginBottom': '12px'}),
            html_table(eps_headers, eps_rows, highlight_row=1),
            dcc.Graph(id='cl-kdist-fig', figure=_CL_KDIST_FIG, config=_GRAPH_CFG,
                      style={'marginTop': '16px'}),
            info_box([
                html.Strong('eps = 0,70 selecionado: '),
                'Melhor equilíbrio entre granularidade (5 clusters) e representatividade (apenas 6,0% de ruído). '
                'Valores menores fragmentam em 10 clusters com 15% de ruído; valores maiores colapsam para 2 grupos.'
            ]),
        ], style=CARD),

        html.Div([
            html.H3('K-Means++, Clusterização Particional', style=H3),
            html.P(
                'O K-Means++ minimiza distância intracluster com inicialização inteligente de '
                'centróides que garante reprodutibilidade. O número K foi determinado pela '
                'análise combinada de três métodos:',
                style=P_STYLE
            ),
            html.Div([
                html.Div([
                    html.Div('Método do Cotovelo', style={'fontWeight': '700', 'color': ROXO, 'marginBottom': '6px'}),
                    html.Div('Identifica o ponto de inflexão na curva de inércia intracluster.', style={'color': TEXT_MUTED, 'fontSize': '13px'}),
                ], style={'padding': '16px', 'backgroundColor': BG_PAGE, 'borderRadius': '8px', 'flex': '1'}),
                html.Div([
                    html.Div('Silhouette Score Médio', style={'fontWeight': '700', 'color': ROXO, 'marginBottom': '6px'}),
                    html.Div('Mede coesão intracluster vs separação intercluster para K ∈ {3, 4, 5}.', style={'color': TEXT_MUTED, 'fontSize': '13px'}),
                ], style={'padding': '16px', 'backgroundColor': BG_PAGE, 'borderRadius': '8px', 'flex': '1'}),
                html.Div([
                    html.Div('Silhouette Diagram', style={'fontWeight': '700', 'color': ROXO, 'marginBottom': '6px'}),
                    html.Div('Visualização individual por cluster para detectar grupos subótimos.', style={'color': TEXT_MUTED, 'fontSize': '13px'}),
                ], style={'padding': '16px', 'backgroundColor': BG_PAGE, 'borderRadius': '8px', 'flex': '1'}),
            ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '20px'}),
            dcc.Graph(id='cl-elbow-sil-fig', figure=_CL_ELBOW_SIL_FIG, config=_GRAPH_CFG,
                      style={'marginTop': '16px'}),
            html.H4('Silhouette Diagram por Cluster (Gráfico 21)', style={'color': ROXO, 'marginTop': '20px', 'marginBottom': '10px'}),
            html.P(
                'O Silhouette Diagram detalha a largura de silhouette individual de cada amostra '
                'agrupada por cluster, permitindo identificar grupos subótimos e confirmar que '
                'K=5 produz separação homogênea sem clusters dominantes ou degenerados.',
                style=P_STYLE
            ),
            dcc.Graph(id='cl-silh-diag-fig', figure=_CL_SILH_DIAG_FIG, config=_GRAPH_CFG,
                      style={'marginTop': '16px'}),
        ], style=CARD),

    ])


def cluster_resultados():
    personas = [
        {
            'id': 0, 'nome': 'O Refém da Metrópole',
            'subtitulo': 'Alto Comprometimento / Vulnerável',
            'vol': '1.938 usuários (9,7%)', 'tier': '1 (Tier_1)',
            'deps': '2,2',
            'aluguel': '29,8%', 'emprest': '13,9%', 'renda_livre': '7,8%',
            'ralos': 'Rent_pct (29,8%) + Loan_Repayment_pct (13,9%)',
            'poupanca': 'Meta não atingida, único cluster com Gap positivo (+2,0%)',
            'color': ERROR,
        },
        {
            'id': 1, 'nome': 'O Equilibrado do Interior',
            'subtitulo': 'Perfil Saudável / Massa Normativa',
            'vol': '8.056 usuários (40,3%)', 'tier': '2/3 (Tier_2/Tier_3)',
            'deps': '1,8',
            'aluguel': '18,4%', 'emprest': '0,9%', 'renda_livre': '33,9%',
            'ralos': 'Rent_pct (18,4%) + Desired_Savings_pct (8,5%)',
            'poupanca': 'Poupa consistentemente acima da meta (Gap: -25,4%)',
            'color': '#0891b2',
        },
        {
            'id': 2, 'nome': 'O Urbano Sem Dívidas',
            'subtitulo': 'Pressão de Aluguel / Sem Endividamento',
            'vol': '3.809 usuários (19,0%)', 'tier': '1 (Tier_1)',
            'deps': '1,9',
            'aluguel': '30,0%', 'emprest': '0,9%', 'renda_livre': '22,0%',
            'ralos': 'Rent_pct (30,0%) + Desired_Savings_pct (9,1%)',
            'poupanca': 'Poupa acima da meta, com folga moderada (Gap: -12,9%)',
            'color': '#6366f1',
        },
        {
            'id': 3, 'nome': 'O Endividado do Interior',
            'subtitulo': 'Pressão de Empréstimos / Risco Moderado',
            'vol': '4.064 usuários (20,3%)', 'tier': '2/3 (Tier_2/Tier_3)',
            'deps': '2,1',
            'aluguel': '18,8%', 'emprest': '13,8%', 'renda_livre': '19,1%',
            'ralos': 'Loan_Repayment_pct (13,8%) + Rent_pct (18,8%)',
            'poupanca': 'Consegue poupar acima da meta, mas com margem reduzida (Gap: -10,0%)',
            'color': '#f59e0b',
        },
        {
            'id': 4, 'nome': 'O Poupador Disciplinado',
            'subtitulo': 'Maior Disciplina Financeira do Grupo',
            'vol': '2.133 usuários (10,7%)', 'tier': '2/3 (Tier_2/Tier_3)',
            'deps': '2,2',
            'aluguel': '20,2%', 'emprest': '2,9%', 'renda_livre': '28,4%',
            'ralos': 'Desired_Savings_pct (17,4%) + Rent_pct (20,2%)',
            'poupanca': 'Maior meta de poupança (17,4%) e cumpre com folga (Gap: -11,0%)',
            'color': SUCCESS,
        },
    ]

    comp_headers = ['Métrica', 'K-Means++', 'DBSCAN']
    comp_rows = [
        ['Silhouette Score', '0,2219 ★', '0,1258'],
        ['Davies-Bouldin',   '1,5041 ★', '1,6867'],
        ['Nº de Clusters',   '5',        '5'],
        ['Ruído Isolado',    'N/A',       '1.208 (6,0%)'],
        ['K Predefinido',    'Sim (K=5)', 'Não'],
        ['Geometria Assumida','Esférica', 'Livre'],
    ]

    persona_cards = []
    for p in personas:
        persona_cards.append(
            html.Div([
                html.Div([
                    html.Div(f'Cluster {p["id"]}', style={
                        'backgroundColor': p['color'], 'color': BRANCO,
                        'borderRadius': '20px', 'padding': '4px 12px',
                        'fontSize': '11px', 'fontWeight': '700',
                        'display': 'inline-block', 'marginBottom': '10px',
                    }),
                    html.Div(p['nome'], style={'fontWeight': '800', 'fontSize': '16px', 'color': TEXT_MAIN, 'marginBottom': '4px'}),
                    html.Div(p['subtitulo'], style={'color': TEXT_MUTED, 'fontSize': '13px', 'marginBottom': '16px'}),
                    html.Div(p['vol'], style={'fontWeight': '700', 'color': p['color'], 'fontSize': '14px', 'marginBottom': '12px'}),
                    html.Div([
                        html.Div([html.Span('City Tier: ', style={'color': TEXT_MUTED, 'fontSize': '12px'}), html.Span(p['tier'], style={'fontWeight': '700', 'fontSize': '13px'})]),
                        html.Div([html.Span('Dependentes: ', style={'color': TEXT_MUTED, 'fontSize': '12px'}), html.Span(p['deps'], style={'fontWeight': '700', 'fontSize': '13px'})]),
                    ], style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginBottom': '12px'}),
                    html.Hr(style={'border': 'none', 'borderTop': f'1px solid {BORDER}', 'margin': '12px 0'}),
                    html.Div([html.Span('Aluguel: ', style={'color': TEXT_MUTED, 'fontSize': '12px'}), html.Span(p['aluguel'], style={'fontWeight': '700', 'color': p['color'], 'fontSize': '13px'})]),
                    html.Div([html.Span('Empréstimos: ', style={'color': TEXT_MUTED, 'fontSize': '12px'}), html.Span(p['emprest'], style={'fontWeight': '700', 'fontSize': '13px'})]),
                    html.Div([html.Span('Renda livre: ', style={'color': TEXT_MUTED, 'fontSize': '12px'}), html.Span(p['renda_livre'], style={'fontWeight': '700', 'fontSize': '13px'})]),
                    html.Div([html.Span('Poupança: ', style={'color': TEXT_MUTED, 'fontSize': '12px'}), html.Span(p['poupanca'], style={'fontWeight': '600', 'fontSize': '12px', 'color': TEXT_MAIN})]),
                    html.Hr(style={'border': 'none', 'borderTop': f'1px solid {BORDER}', 'margin': '12px 0'}),
                    html.Div('Top 2 ralos financeiros:', style={'fontSize': '11px', 'color': TEXT_MUTED, 'fontWeight': '600', 'textTransform': 'uppercase', 'marginBottom': '4px'}),
                    html.Div(p['ralos'], style={'fontSize': '13px', 'color': TEXT_MAIN, 'fontWeight': '600'}),
                ]),
            ], style={
                'backgroundColor': BRANCO, 'borderRadius': '12px', 'padding': '20px',
                'border': f'1px solid {BORDER}', 'boxShadow': '0 2px 8px rgba(0,0,0,0.05)',
                'borderTop': f'4px solid {p["color"]}',
            })
        )

    return html.Div([
        kdd_tag('Resultados'),
        html.H2('Resultados e Análise do Modelo de Clusterização', style=H2),

        # DBSCAN
        html.Div([
            html.H3('7.2.1, Resultados DBSCAN (eps=0,70, min_samples=12)', style=H3),
            html.P(
                'Os parâmetros finais produziram 5 clusters estruturais com 1.208 pontos isolados '
                'como ruído (6,0% da base). O Silhouette Score de 0,1258 deve ser interpretado com '
                'cautela: essa métrica penaliza por design clusters de geometria não esférica,'
                'premissa incompatível com o DBSCAN.',
                style=P_STYLE
            ),
            html.Div([
                metric_card('Silhouette Score', '0,1258', '#0891b2'),
                metric_card('Davies-Bouldin',   '1,6867', '#6366f1'),
                metric_card('Clusters Reais',   '5',       ROXO),
                metric_card('Ruído (noise)',     '6,0%',    TEXT_MUTED),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))', 'gap': '14px', 'marginBottom': '20px'}),
            info_box([
                html.Strong('Discriminadores primários (SHAP + Árvore de Decisão): '),
                'Gap_Poupanca, Disposable_pct, Rent_pct e City_Tier_enc. '
                'Gastos_Consumo_pct e Gastos_Fixos_pct apresentam importância próxima de zero,'
                'o padrão de consumo básico é homogêneo entre os perfis.'
            ]),
        ], style=CARD),

        # K-Means++
        html.Div([
            html.H3('7.2.2, Resultados K-Means++ (K=5)', style={**H3, 'color': SUCCESS}),
            html.P(
                'K=5 foi selecionado com base no Silhouette Diagram mais equilibrado e maior '
                'Silhouette médio (0,217) do intervalo avaliado. O Davies-Bouldin melhora de '
                '1,77 (K=3) para 1,50 (K=5). O K-Means++ superou os demais algoritmos em '
                'ambas as métricas.',
                style=P_STYLE
            ),
            html.Div([
                metric_card('Silhouette Score', '0,2219', SUCCESS),
                metric_card('Davies-Bouldin',   '1,5041', ROXO),
                metric_card('Clusters',         '5',       '#6366f1'),
                metric_card('K Predefinido',    'Sim',     '#0891b2'),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))', 'gap': '14px', 'marginBottom': '24px'}),

            html.H3('Visualização 2D, Projeção PCA dos Clusters (Gráfico 27)', style={**H3, 'marginTop': '8px'}),
            html.P(
                'A projeção no espaço PCA bidimensional revela regiões de concentração distintas, '
                'com sobreposição entre clusters adjacentes. Esta visualização confirma que K=5 '
                'captura estruturas reais nos dados, não artefatos do algoritmo.',
                style=P_STYLE
            ),
            dcc.Graph(id='cl-pca2d-fig', figure=_CL_PCA2D_FIG, config=_GRAPH_CFG,
                      style={'marginTop': '16px'}),

            html.H3('Interpretabilidade, Modelo Surrogate (Random Forest + SHAP)', style={**H3, 'marginTop': '24px'}),
            html.P(
                'A interpretabilidade do K-Means++ foi obtida via modelo surrogate treinado para reproduzir '
                'os rótulos de cluster. O Random Forest com SHAP Values quantifica a contribuição global '
                'de cada atributo na separação entre as 5 personas.',
                style=P_STYLE
            ),

            html.Div([
                html.Div([
                    html.H4('SHAP Global, Importância das Variáveis (Gráfico 30)', style={'color': ROXO, 'marginBottom': '10px', 'fontSize': '15px'}),
                    html.P(
                        'Gap_Poupanca, Disposable_pct, Rent_pct, Loan_Repayment_pct e City_Tier_enc '
                        'são as variáveis discriminantes de maior impacto. As 5 personas se diferenciam '
                        'principalmente nas dimensões de folga de renda e comprometimento com aluguel e dívida.',
                        style={**P_STYLE, 'marginBottom': '12px'}
                    ),
                    dcc.Graph(id='cl-shap-fig', figure=_CL_SHAP_FIG, config=_GRAPH_CFG),
                ], style={'flex': '1', 'minWidth': '280px'}),

                html.Div([
                    html.H4('Princípio de Pareto, Comprometimento por Cluster (Gráfico 29)', style={'color': ROXO, 'marginBottom': '10px', 'fontSize': '15px'}),
                    html.P(
                        'Rent_pct é a categoria de maior peso em quatro dos cinco clusters. '
                        'A exceção é o Cluster 3 (O Poupador Agressivo), onde Loan_Repayment_pct assume '
                        'a posição dominante, reforçando a separação comportamental entre os perfis.',
                        style={**P_STYLE, 'marginBottom': '12px'}
                    ),
                    dcc.Graph(id='cl-pareto-fig', figure=_CL_PARETO_FIG, config=_GRAPH_CFG),
                ], style={'flex': '1', 'minWidth': '280px'}),
            ], style={'display': 'flex', 'gap': '28px', 'flexWrap': 'wrap', 'marginTop': '8px'}),
        ], style=CARD),

        # Comparativo
        html.Div([
            html.H3('7.2.3, Comparativo K-Means++ vs DBSCAN (Tabela 8)', style=H3),
            html.P(
                'Ambos convergem para 5 segmentos por mecanismos completamente distintos, '
                'validando a estrutura de 5 personas como característica intrínseca do dataset.',
                style=P_STYLE
            ),
            html_table(comp_headers, comp_rows, highlight_row=0),
            info_box([
                html.Strong('Diferença-chave: '),
                'O K-Means++ isolou o perfil de alta vulnerabilidade (Cluster 0) como um grupo '
                'bem definido, enquanto o DBSCAN dispersou esses casos entre o ruído. '
                'A convergência para 5 segmentos por paradigmas distintos valida a estrutura.'
            ], border_color=SUCCESS),
        ], style=CARD),

        # 5 Personas
        html.Div([
            html.Div([badge('K-MEANS++ · 5 PERSONAS')], style={'marginBottom': '12px'}),
            html.H3('Personas Identificadas', style={**H3, 'marginTop': '4px', 'fontSize': '20px'}),
            html.P(
                'O K-Means++ identificou 5 perfis financeiros comportamentais distintos. '
                'Em combinação com o modelo de classificação — que foca em prever clientes seguros '
                'com bom potencial de crédito — é possível não só identificar o perfil comportamental '
                'de cada cliente, mas também priorizar concessões de crédito com confiança.',
                style=P_STYLE
            ),
            html.Div(persona_cards,
                     style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(280px, 1fr))', 'gap': '20px'}),
        ], style=CARD),

        # Integração
        html.Div([
            html.H3('7.2.4, Integração com o Modelo de Classificação', style=H3),
            html.P(
                'Os dois modelos formam um sistema em duas camadas:',
                style=P_STYLE
            ),
            html.Div([
                html.Div([
                    html.Div('Clusterização', style={'fontWeight': '700', 'color': '#7c3aed', 'marginBottom': '6px', 'fontSize': '15px'}),
                    html.Div('Identifica qual é o perfil financeiro e por qual mecanismo o risco se origina.', style={'color': TEXT_MUTED, 'fontSize': '14px'}),
                ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'rgba(124,58,237,0.06)',
                          'borderRadius': '8px', 'borderLeft': '4px solid #7c3aed'}),
                html.Div('+', style={'fontSize': '28px', 'fontWeight': '300', 'color': TEXT_MUTED,
                                      'display': 'flex', 'alignItems': 'center', 'padding': '0 8px'}),
                html.Div([
                    html.Div('Classificação', style={'fontWeight': '700', 'color': ROXO, 'marginBottom': '6px', 'fontSize': '15px'}),
                    html.Div('Prevê quais clientes são seguros e têm potencial de crédito favorável — foco em identificar bons pagadores, não apenas evitar maus.', style={'color': TEXT_MUTED, 'fontSize': '14px'}),
                ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'rgba(29,18,82,0.06)',
                          'borderRadius': '8px', 'borderLeft': '4px solid ' + ROXO}),
                html.Div('=', style={'fontSize': '28px', 'fontWeight': '300', 'color': TEXT_MUTED,
                                      'display': 'flex', 'alignItems': 'center', 'padding': '0 8px'}),
                html.Div([
                    html.Div('Sistema de Crédito Comportamental', style={'fontWeight': '700', 'color': SUCCESS, 'marginBottom': '6px', 'fontSize': '15px'}),
                    html.Div('Intervenções diferenciadas por perfil, algo impossível com apenas um dos modelos.', style={'color': TEXT_MUTED, 'fontSize': '14px'}),
                ], style={'flex': '1', 'padding': '20px', 'backgroundColor': 'rgba(46,125,50,0.06)',
                          'borderRadius': '8px', 'borderLeft': f'4px solid {SUCCESS}'}),
            ], style={'display': 'flex', 'gap': '12px', 'alignItems': 'center', 'flexWrap': 'wrap'}),
        ], style=CARD),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def _sidebar(nav_id, items):
    return html.Div([
        html.H3('Etapas KDD', style={
            'marginTop': '0', 'fontSize': '14px', 'color': TEXT_MUTED,
            'fontWeight': '700', 'textTransform': 'uppercase',
            'letterSpacing': '1px', 'marginBottom': '20px',
        }),
        dcc.RadioItems(
            id=nav_id,
            options=[{'label': v, 'value': k} for k, v in items],
            value=items[0][0],
            className='custom-sidebar-menu',
            labelStyle={
                'display': 'flex', 'alignItems': 'center', 'cursor': 'pointer',
                'padding': '11px 16px', 'marginBottom': '6px',
                'borderRadius': '6px', 'fontSize': '14px',
                'color': TEXT_MUTED, 'transition': 'all 0.2s ease',
                'fontWeight': '500',
            },
            inputStyle={'marginRight': '10px', 'cursor': 'pointer'},
        ),
    ], style={
        'width': '220px', 'flexShrink': '0',
        'backgroundColor': BRANCO,
        'borderRadius': '14px',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.05)',
        'padding': '28px 20px',
        'height': 'fit-content',
    })


CLASSIF_STEPS = [
    ('proc',    'Processamento'),
    ('transf',  'Transformação'),
    ('miner',   'Mineração'),
    ('result',  'Resultados'),
]

CLUSTER_STEPS = [
    ('proc',    'Processamento'),
    ('transf',  'Transformação'),
    ('miner',   'Mineração'),
    ('result',  'Resultados'),
]

layout = html.Div([

    # ── Cabeçalho da seção ────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1('Machine Learning', style={
                'color': TEXT_MAIN, 'fontSize': '28px', 'fontWeight': '800',
                'margin': '0 0 6px 0',
            }),
            html.P(
                'Modelos supervisionado e não supervisionado aplicados à base financeira,'
                'estruturados segundo a metodologia KDD.',
                style={'color': TEXT_MUTED, 'fontSize': '15px', 'margin': '0'},
            ),
        ]),
    ], style={
        'padding': '32px 40px 24px 40px',
        'borderBottom': f'1px solid {BORDER}',
        'backgroundColor': BRANCO,
    }),

    # ── Tabs de modelo ────────────────────────────────────────────────────────
    html.Div([
        dcc.Tabs(
            id='ml-model-tabs',
            value='classif',
            children=[
                dcc.Tab(label='Classificação, Supervisionado',    value='classif',
                        className='custom-tab', selected_className='custom-tab-selected'),
                dcc.Tab(label='Clusterização, Não Supervisionado', value='cluster',
                        className='custom-tab', selected_className='custom-tab-selected'),
            ],
            style={'borderBottom': 'none'},
        ),
    ], style={
        'padding': '0 40px',
        'backgroundColor': BRANCO,
        'borderBottom': f'1px solid {BORDER}',
    }),

    # ── Área de conteúdo ──────────────────────────────────────────────────────
    html.Div(id='ml-tab-content', style={
        'padding': '32px 40px',
        'backgroundColor': BG_PAGE,
        'minHeight': 'calc(100vh - 180px)',
    }),

], style={
    'fontFamily': '"Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    'backgroundColor': BG_PAGE,
})


# ══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════

def register_ml_callbacks(app):

    # ── Renderiza conteúdo da tab selecionada ─────────────────────────────────
    @app.callback(
        Output('ml-tab-content', 'children'),
        Input('ml-model-tabs', 'value'),
    )
    def render_tab(tab):
        sidebar_classif = _sidebar('ml-classif-nav', CLASSIF_STEPS)
        sidebar_cluster = _sidebar('ml-cluster-nav', CLUSTER_STEPS)

        if tab == 'classif':
            return html.Div([
                sidebar_classif,
                html.Div([
                    html.Div(classif_processamento(), id='ml-c-proc'),
                    html.Div(classif_transformacao(), id='ml-c-transf', style={'display': 'none'}),
                    html.Div(classif_mineracao(),     id='ml-c-miner',  style={'display': 'none'}),
                    html.Div(classif_resultados(),    id='ml-c-result', style={'display': 'none'}),
                ], style={'flex': '1', 'minWidth': '0'}),
            ], style={'display': 'flex', 'gap': '28px', 'alignItems': 'flex-start'})

        else:
            return html.Div([
                sidebar_cluster,
                html.Div([
                    html.Div(cluster_processamento(), id='ml-k-proc'),
                    html.Div(cluster_transformacao(), id='ml-k-transf', style={'display': 'none'}),
                    html.Div(cluster_mineracao(),     id='ml-k-miner',  style={'display': 'none'}),
                    html.Div(cluster_resultados(),    id='ml-k-result', style={'display': 'none'}),
                ], style={'flex': '1', 'minWidth': '0'}),
            ], style={'display': 'flex', 'gap': '28px', 'alignItems': 'flex-start'})

    # ── Navegação sidebar Classificação ───────────────────────────────────────
    @app.callback(
        [Output('ml-c-proc',   'style'), Output('ml-c-transf', 'style'),
         Output('ml-c-miner',  'style'), Output('ml-c-result', 'style')],
        Input('ml-classif-nav', 'value'),
        prevent_initial_call=True,
    )
    def toggle_classif(sel):
        keys = ['proc', 'transf', 'miner', 'result']
        return tuple({'display': 'block'} if sel == k else {'display': 'none'} for k in keys)

    # ── Navegação sidebar Clusterização ──────────────────────────────────────
    @app.callback(
        [Output('ml-k-proc',   'style'), Output('ml-k-transf', 'style'),
         Output('ml-k-miner',  'style'), Output('ml-k-result', 'style')],
        Input('ml-cluster-nav', 'value'),
        prevent_initial_call=True,
    )
    def toggle_cluster(sel):
        keys = ['proc', 'transf', 'miner', 'result']
        return tuple({'display': 'block'} if sel == k else {'display': 'none'} for k in keys)

    # ── Helpers compartilhados ────────────────────────────────────────────────
    def _load_model():
        """
        Carrega o payload numpy puro (sem sklearn em runtime).
        Novo modelo: LogisticRegression direta sobre rácios, sem StandardScaler.
        Retorna: (coef, intercept, features)
        """
        p = joblib.load(MODEL_PATH)
        coef      = np.array(p['coef'])       # (1, 9)
        intercept = np.array(p['intercept'])  # (1,)
        return coef, intercept, p['features']

    def _predict_proba(X_arr, coef, intercept):
        """Regressão Logística — sigmoid puro numpy (sem scaler)."""
        z = X_arr @ coef.T + intercept   # (n, 1)
        return 1.0 / (1.0 + np.exp(-z)) # prob da classe 1 (Vulnerável)

    def _build_ratios(df):
        """Constrói as 9 features de rácios a partir do DataFrame bruto."""
        df = df.copy()
        df['Rent_Ratio']       = df['Rent']       / df['Income']
        df['Healthcare_Ratio'] = df['Healthcare'] / df['Income']
        df['Education_Ratio']  = df['Education']  / df['Income']
        df['Groceries_Ratio']  = df['Groceries']  / df['Income']
        df['Transport_Ratio']  = df['Transport']  / df['Income']
        df['Utilities_Ratio']  = df['Utilities']  / df['Income']
        df['Insurance_Ratio']  = df['Insurance']  / df['Income']
        return df

    def _load_base():
        """Carrega dataset, filtra Desired_Savings > 0 → ~19.888 registros.
        Tenta parquet primeiro; cai para CSV se engine indisponível."""
        try:
            df = pd.read_parquet(DATA_PATH, engine='fastparquet')
        except Exception:
            try:
                df = pd.read_parquet(DATA_PATH)
            except Exception:
                df = pd.read_csv(DATA_PATH.replace('.parquet', '.csv'))
        df = df[df['Desired_Savings'] > 0].reset_index(drop=True)
        return _build_ratios(df)

    def _nivel_seguro(p_seguro):
        """Classifica o perfil de crédito pelo p(Seguro) = 1 - p(Vulnerável)."""
        if p_seguro >= 0.70: return 'Alta confiança'
        if p_seguro >= 0.40: return 'Moderada confiança'
        return 'Baixa confiança'

    def _nivel(p_vuln):
        """Mantido para compatibilidade com a demo (usa p(Vulnerável))."""
        if p_vuln < 0.30: return 'Baixo'
        if p_vuln < 0.60: return 'Médio'
        return 'Alto'

    def _result_card(label, value, color):
        return html.Div([
            html.Div(label, style={'color': '#666', 'fontSize': '12px', 'fontWeight': '600',
                                    'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
            html.Div(value, style={'color': color, 'fontSize': '22px', 'fontWeight': 'bold', 'marginTop': '4px'}),
        ], style={
            'backgroundColor': BRANCO, 'padding': '18px', 'borderRadius': '10px',
            'boxShadow': '0 2px 6px rgba(0,0,0,0.05)', 'borderLeft': f'4px solid {color}',
        })

    # ── Atualiza valor exibido do threshold em tempo real ────────────────────
    @app.callback(
        Output('ml-threshold-val', 'children'),
        Input('ml-threshold', 'value'),
    )
    def _update_threshold_label(val):
        return f'{(val or 0.5):.2f}'

    # ── Demo na Base Completa ─────────────────────────────────────────────────
    @app.callback(
        Output('ml-pie',        'figure'),
        Output('ml-hist',       'figure'),
        Output('ml-demo-cards', 'children'),
        Output('ml-store',      'data'),
        Input('ml-run-btn',     'n_clicks'),
        State('ml-threshold',   'value'),
        prevent_initial_call=True,
    )
    def run_demo(n_clicks, threshold):
        try:
            coef, intercept, features = _load_model()
            df = _load_base()

            faltando = [c for c in features if c not in df.columns]
            if faltando:
                empty = go.Figure()
                err = html.Div(f'Colunas ausentes: {", ".join(faltando)}', style={'color': ERROR})
                return empty, empty, err, {}

            X             = df[features].fillna(0).to_numpy(dtype=float)
            prob_vuln     = _predict_proba(X, coef, intercept).ravel()  # p(Vulnerável)
            df['prob_seguro'] = 1.0 - prob_vuln
            df['prob_risco']  = prob_vuln

        except Exception as exc:
            empty = go.Figure()
            err = html.Div(f'Erro ao carregar o modelo: {exc}', style={'color': ERROR, 'fontSize': '14px'})
            return empty, empty, err, {}

        # threshold aplicado sobre p(Vulnerável): acima do limiar → Vulnerável
        df['classif'] = (df['prob_risco'] >= threshold).astype(int)
        df['nivel']   = df['prob_risco'].apply(_nivel)

        # Gráfico de pizza — proporção Seguro × Vulnerável com o threshold atual
        classif_labels = df['classif'].map({0: 'Seguro', 1: 'Vulnerável'})
        pie = px.pie(
            classif_labels.rename('Classificação').to_frame(),
            names='Classificação',
            title=f'Seguro × Vulnerável (threshold vulnerável = {threshold:.2f})',
            color='Classificação',
            color_discrete_map={'Seguro': SUCCESS, 'Vulnerável': ERROR},
            hole=0.4,
        )
        pie.update_traces(textposition='inside', textinfo='percent+label')
        pie.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          showlegend=False, font_family='Segoe UI',
                          margin=dict(t=50, b=20, l=20, r=20))

        # Histograma de p(Vulnerável)
        hist = px.histogram(df, x='prob_risco', nbins=50,
                            title='Distribuição da Probabilidade de Ser Vulnerável',
                            labels={'prob_risco': 'Probabilidade de Ser Vulnerável'},
                            color_discrete_sequence=[ERROR])
        hist.add_vline(x=threshold, line_dash='dash', line_color=ROXO, line_width=2,
                       annotation_text=f'Threshold: {threshold:.2f}',
                       annotation_position='top right',
                       annotation_font=dict(color=ROXO, size=11))
        hist.add_vline(x=0.70, line_dash='dot', line_color=ERROR,
                       annotation_text='Alta (≥ 70%)',
                       annotation_position='top left',
                       annotation_font=dict(color=ERROR, size=10))
        hist.add_vline(x=0.40, line_dash='dot', line_color='#f59e0b',
                       annotation_text='Moderada (≥ 40%)',
                       annotation_position='top left',
                       annotation_font=dict(color='#f59e0b', size=10))
        hist.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                           yaxis_title='Nº de Clientes', font_family='Segoe UI',
                           margin=dict(t=50, b=40, l=40, r=20))
        hist.update_xaxes(showgrid=False)
        hist.update_yaxes(gridcolor='#f0f0f0')

        total    = len(df)
        seguros  = (df['classif'] == 0).sum()
        vuln     = (df['classif'] == 1).sum()
        # clientes com baixa probabilidade de vulnerabilidade (< 0.40) = alta confiança de segurança
        alta_seg = (df['prob_risco'] < 0.40).sum()

        cards = html.Div([
            _result_card('Registros Analisados',    f'{total:,}',    ROXO),
            _result_card('Seguros',                 f'{seguros:,}',  SUCCESS),
            _result_card('Vulneráveis',             f'{vuln:,}',     ERROR),
            _result_card('Baixa Vulnerab. (< 40%)', f'{alta_seg:,}', '#6366f1'),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(180px, 1fr))',
                  'gap': '14px', 'marginBottom': '20px'})

        return (
            pie, hist, cards,
            df[['prob_seguro', 'prob_risco', 'classif', 'nivel']].to_json(orient='records', force_ascii=False),
        )

    # ── Botões +/− dos campos de previsão individual (pattern matching) ─────────
    @app.callback(
        Output({'type': 'ip-num',   'id': MATCH}, 'value'),
        Input({'type':  'ip-plus',  'id': MATCH}, 'n_clicks'),
        Input({'type':  'ip-minus', 'id': MATCH}, 'n_clicks'),
        State({'type':  'ip-num',   'id': MATCH}, 'value'),
        prevent_initial_call=True,
    )
    def _ajustar_campo(n_plus, n_minus, atual):
        tid = ctx.triggered_id
        if not tid:
            return dash.no_update
        passo = 5 if tid['id'] in ('age', 'dependents') else 50
        delta = passo if tid['type'] == 'ip-plus' else -passo
        try:
            val_num = float(str(atual).strip()) if atual not in (None, '', 0) else 0.0
        except (ValueError, TypeError):
            val_num = 0.0
        resultado = max(0, val_num + delta)
        return str(int(resultado))

    # ── Previsão Individual ───────────────────────────────────────────────────
    @app.callback(
        Output('ip-predict-result', 'children'),
        Input('ip-predict-btn', 'n_clicks'),
        State({'type': 'ip-num', 'id': ALL}, 'value'),
        prevent_initial_call=True,
    )
    def run_individual_pred(n_clicks, all_vals):
        # Mapeia valores pelo id do campo via ctx.states_list (converte string→float)
        def _to_float(v):
            try:
                return float(str(v).strip()) if v not in (None, '') else 0.0
            except (ValueError, TypeError):
                return 0.0
        field_map = {item['id']['id']: _to_float(item['value']) for item in ctx.states_list[0]}

        vals = {
            'Income':            field_map.get('income', 0),
            'Age':               field_map.get('age', 0),
            'Dependents':  field_map.get('dependents', 0),
            'Healthcare':  field_map.get('healthcare', 0),
            'Rent':        field_map.get('rent', 0),
            'Groceries':   field_map.get('groceries', 0),
            'Education':   field_map.get('education', 0),
            'Transport':   field_map.get('transport', 0),
            'Utilities':   field_map.get('utilities', 0),
            'Insurance':   field_map.get('insurance', 0),
        }

        inc = vals['Income']
        if inc <= 0:
            return html.Div('⚠️ Renda deve ser maior que zero.', style={'color': ERROR, 'fontWeight': '600'})

        try:
            coef, intercept, features = _load_model()
        except Exception as exc:
            return html.Div(f'Erro ao carregar modelo: {exc}', style={'color': ERROR})

        # Construir rácios — exactamente as 9 features do modelo
        ratios = {
            'Age':               vals['Age'],
            'Dependents':        vals['Dependents'],
            'Rent_Ratio':        vals['Rent']       / inc,
            'Healthcare_Ratio':  vals['Healthcare'] / inc,
            'Education_Ratio':   vals['Education']  / inc,
            'Groceries_Ratio':   vals['Groceries']  / inc,
            'Transport_Ratio':   vals['Transport']  / inc,
            'Utilities_Ratio':   vals['Utilities']  / inc,
            'Insurance_Ratio':   vals['Insurance']  / inc,
        }

        X_row     = np.array([[ratios[f] for f in features]], dtype=float)
        prob_vuln = float(_predict_proba(X_row, coef, intercept)[0, 0])

        # Classificação por confiança — thresholds sobre p(Vulnerável)
        if prob_vuln >= 0.70:
            perfil_label = 'ALTA CONFIANÇA DE VULNERABILIDADE'
            gauge_color  = ERROR
            perfil_desc  = (f'{prob_vuln*100:.1f}% de probabilidade de ser Vulnerável. '
                            'Crédito não recomendado sem garantias adicionais.')
        elif prob_vuln >= 0.40:
            perfil_label = 'MODERADA CONFIANÇA DE VULNERABILIDADE'
            gauge_color  = '#f59e0b'
            perfil_desc  = (f'{prob_vuln*100:.1f}% de probabilidade de ser Vulnerável. '
                            'Análise complementar recomendada antes de conceder crédito.')
        else:
            perfil_label = 'BAIXA CONFIANÇA DE VULNERABILIDADE'
            gauge_color  = SUCCESS
            perfil_desc  = (f'{prob_vuln*100:.1f}% de probabilidade de ser Vulnerável. '
                            'Perfil favorável — candidato elegível para concessão de crédito.')

        # Classificação binária (threshold 0,50 padrão)
        is_vuln     = prob_vuln >= 0.50
        class_color = ERROR if is_vuln else SUCCESS
        class_label = 'VULNERÁVEL' if is_vuln else 'SEGURO'

        # Gauge de p(Vulnerável)
        gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=round(prob_vuln * 100, 1),
            number={'suffix': '%', 'font': {'size': 32, 'color': gauge_color}},
            gauge={
                'axis': {'range': [0, 100], 'ticksuffix': '%'},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0,  40],  'color': 'rgba(46,125,50,0.15)'},
                    {'range': [40, 70],  'color': 'rgba(245,158,11,0.15)'},
                    {'range': [70, 100], 'color': 'rgba(229,75,75,0.15)'},
                ],
                'threshold': {'line': {'color': gauge_color, 'width': 3}, 'value': prob_vuln * 100},
            },
            title={'text': 'Probabilidade de Ser Vulnerável', 'font': {'size': 14, 'color': TEXT_MUTED}},
        ))
        gauge.update_layout(
            height=260, paper_bgcolor='white', font_family='Segoe UI',
            margin=dict(t=40, b=10, l=20, r=20),
        )

        # Tabela de rácios usados pelo modelo
        ratio_rows = [
            ('Age',            f'{ratios["Age"]:.0f}',                    'anos'),
            ('Dependents',     f'{ratios["Dependents"]:.0f}',             'pessoas'),
            ('Rent_Ratio',     f'{ratios["Rent_Ratio"]*100:.1f}%',        'Aluguel / Renda'),
            ('Healthcare_Ratio',f'{ratios["Healthcare_Ratio"]*100:.1f}%', 'Saúde / Renda'),
            ('Education_Ratio', f'{ratios["Education_Ratio"]*100:.1f}%',  'Educação / Renda'),
            ('Groceries_Ratio', f'{ratios["Groceries_Ratio"]*100:.1f}%',  'Supermercado / Renda'),
            ('Transport_Ratio', f'{ratios["Transport_Ratio"]*100:.1f}%',  'Transporte / Renda'),
            ('Utilities_Ratio', f'{ratios["Utilities_Ratio"]*100:.1f}%',  'Utilidades / Renda'),
            ('Insurance_Ratio', f'{ratios["Insurance_Ratio"]*100:.1f}%',  'Seguro / Renda'),
        ]
        ratio_items = [
            html.Div([
                html.Span(feat, style={'fontWeight': '600', 'fontSize': '12px',
                                       'color': ROXO, 'minWidth': '150px', 'display': 'inline-block'}),
                html.Span(val,  style={'fontWeight': '700', 'fontSize': '13px',
                                       'color': TEXT_MAIN, 'marginRight': '8px'}),
                html.Span(desc, style={'color': TEXT_MUTED, 'fontSize': '11px'}),
            ], style={'padding': '6px 0', 'borderBottom': f'1px solid {BORDER}'})
            for feat, val, desc in ratio_rows
        ]

        _sec_label = lambda txt: html.Div(txt, style={
            'fontSize': '11px', 'color': TEXT_MUTED, 'fontWeight': '700',
            'textTransform': 'uppercase', 'letterSpacing': '0.8px', 'marginBottom': '12px',
        })

        return html.Div([
            html.Div([
                # Bloco 1 — Gauge p(Vulnerável)
                html.Div([
                    _sec_label('Resultado — Regressão Logística'),
                    html.Div(dcc.Graph(figure=gauge, config={'displayModeBar': False})),
                    html.Div(html.Span(perfil_label, style={
                        'backgroundColor': gauge_color, 'color': BRANCO,
                        'borderRadius': '6px', 'padding': '6px 18px',
                        'fontSize': '12px', 'fontWeight': '800', 'letterSpacing': '0.8px',
                        'display': 'inline-block',
                    }), style={'textAlign': 'center', 'marginTop': '8px'}),
                    html.P(perfil_desc,
                           style={'color': TEXT_MUTED, 'fontSize': '12px', 'lineHeight': '1.6',
                                  'marginTop': '10px', 'marginBottom': '8px'}),
                    html.Div([
                        html.Span('Classificação (threshold 0,50): ',
                                  style={'color': TEXT_MUTED, 'fontSize': '12px'}),
                        html.Span(class_label, style={
                            'backgroundColor': class_color, 'color': BRANCO,
                            'borderRadius': '4px', 'padding': '2px 10px',
                            'fontSize': '12px', 'fontWeight': '800',
                        }),
                    ]),
                ], style={
                    'flex': '1', 'minWidth': '280px',
                    'backgroundColor': BG_PAGE, 'padding': '20px', 'borderRadius': '10px',
                    'border': f'1px solid {BORDER}',
                }),

                # Bloco 2 — Rácios inseridos no modelo
                html.Div([
                    _sec_label('Features Fornecidas ao Modelo (9 variáveis)'),
                    html.P('Valores convertidos em rácios sobre a renda antes de entrar no modelo:',
                           style={'color': TEXT_MUTED, 'fontSize': '12px', 'marginBottom': '12px'}),
                    *ratio_items,
                ], style={
                    'flex': '1', 'minWidth': '280px',
                    'backgroundColor': BG_PAGE, 'padding': '20px', 'borderRadius': '10px',
                    'border': f'1px solid {BORDER}',
                }),
            ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}),
        ])
