import dash
from dash import dcc, html, Input, Output, State, MATCH, ALL, ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os

# ─── Caminhos ────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(ROOT_DIR, '..', 'modelos', 'modelo_previsao_vulnerabilidade', 'modelo_regressao_logistica.pkl')
DATA_PATH  = os.path.join(ROOT_DIR, 'dataset', 'data.parquet')

FEATURES = [
    'Income', 'Age', 'Dependents', 'Loan_Repayment',
    'Eating_Out', 'Entertainment', 'Healthcare',
    'Rent', 'Groceries', 'Disposable_Income', 'Desired_Savings'
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
    Gera os 4 gráficos interativos do modelo usando apenas pandas/numpy/plotly.
    Retorna: (corr_fig, cm_fig, shap_fig, pareto_fig)
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
        p       = joblib.load(MODEL_PATH)
        coef    = np.array(p['coef']).ravel()       # (11,)
        intcpt  = float(np.array(p['intercept']).ravel()[0])
        mean_   = np.array(p['scaler_mean'])        # (11,)
        scale_  = np.array(p['scaler_scale'])       # (11,)

        # ── Carregar e preparar dados ─────────────────────────────────────
        try:
            df = pd.read_parquet(DATA_PATH)
        except Exception:
            df = pd.read_csv(DATA_PATH.replace('.parquet', '.csv'))
        df = df[df['Desired_Savings'] > 0].reset_index(drop=True)

        X   = df[FEATURES]
        y   = df['Vulnerable'].to_numpy() if 'Vulnerable' in df.columns else None

        # Reconstruir y se não estiver na base
        if y is None:
            colunas_pot = [
                'Potential_Savings_Groceries', 'Potential_Savings_Transport',
                'Potential_Savings_Eating_Out', 'Potential_Savings_Entertainment',
                'Potential_Savings_Utilities', 'Potential_Savings_Healthcare',
                'Potential_Savings_Education', 'Potential_Savings_Miscellaneous',
            ]
            df['perc_nao_essenciais']    = (df['Eating_Out'] + df['Entertainment']) / df['Income']
            df['perc_emprestimo']        = df['Loan_Repayment'] / df['Income']
            df['perc_potential_savings'] = df[colunas_pot].sum(axis=1) / df['Income']
            df['buffer_emergencia']      = (df['Disposable_Income'] - df['Desired_Savings']) / df['Income']
            df['risk_score'] = (
                (df['perc_emprestimo']      > 0.10).astype(int) +
                (df['perc_nao_essenciais']  > 0.085).astype(int) +
                (df['buffer_emergencia']    < 0.10).astype(int) +
                (df['perc_potential_savings'] > 0.08).astype(int)
            )
            y = (df['risk_score'] >= 2).astype(int).to_numpy()

        X_arr = X.to_numpy(dtype=float)

        # ── 1. Matriz de Correlação ───────────────────────────────────────
        corr       = X.corr().round(2)
        feat_labels = [f.replace('_', ' ') for f in FEATURES]

        corr_fig = px.imshow(
            corr.values,
            x=feat_labels, y=feat_labels,
            text_auto='.2f',
            color_continuous_scale='RdYlGn',
            range_color=[-1, 1],
            title='Gráfico 1 — Matriz de Correlação das 11 Features Preditoras',
        )
        corr_fig.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            height=520, margin=dict(t=60, b=80, l=140, r=40),
            font=dict(size=11, family='Segoe UI'),
            title_font_size=13,
            coloraxis_colorbar=dict(title='r'),
        )
        corr_fig.update_xaxes(tickangle=-40)
        corr_fig.update_traces(
            hovertemplate='<b>%{y}</b> ↔ <b>%{x}</b><br>r = %{z:.2f}<extra></extra>'
        )

        # ── Split estratificado (sem sklearn) ─────────────────────────────
        rng = np.random.RandomState(42)
        test_idx = []
        for cls in np.unique(y):
            cls_idx  = np.where(y == cls)[0]
            shuffled = rng.permutation(cls_idx)
            n_test   = int(np.round(len(cls_idx) * 0.2))
            test_idx.extend(shuffled[:n_test].tolist())
        test_idx  = np.array(test_idx)
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)

        X_test_arr = X_arr[test_idx]
        y_test     = y[test_idx]
        X_test_sc  = (X_test_arr - mean_) / scale_

        z_test   = X_test_sc @ coef + intcpt
        prob_test = 1.0 / (1.0 + np.exp(-z_test))
        y_pred   = (prob_test >= 0.5).astype(int)

        # ── 2. Matriz de Confusão ─────────────────────────────────────────
        tn = int(((y_test == 0) & (y_pred == 0)).sum())
        fp = int(((y_test == 0) & (y_pred == 1)).sum())
        fn = int(((y_test == 1) & (y_pred == 0)).sum())
        tp = int(((y_test == 1) & (y_pred == 1)).sum())
        recall_vuln = tp / (tp + fn) if (tp + fn) > 0 else 0

        lbs    = ['Seguro (0)', 'Vulnerável (1)']
        cm_z   = [[tn, fp], [fn, tp]]
        cm_txt = [[f'TN = {tn:,}', f'FP = {fp:,}'], [f'FN = {fn:,}', f'TP = {tp:,}']]

        cm_fig = go.Figure(go.Heatmap(
            z=cm_z, x=lbs, y=lbs,
            text=cm_txt, texttemplate='<b>%{text}</b>',
            colorscale='Blues', showscale=True,
            hovertemplate='Real: %{y}<br>Previsto: %{x}<br>%{text}<extra></extra>',
        ))
        cm_fig.update_layout(
            title='Figura 17 — Matriz de Confusão, Regressão Logística',
            xaxis_title='Previsto pelo Modelo',
            yaxis_title='Rótulo Real',
            paper_bgcolor='white', plot_bgcolor='white',
            height=420, font=dict(size=12, family='Segoe UI'),
            title_font_size=13,
            margin=dict(t=60, b=80, l=120, r=40),
            annotations=[dict(
                text=f'Recall Vulnerável: {recall_vuln*100:.1f}%   |   '
                     f'AUC-ROC: {p.get("auc_roc", "0.9506")}',
                x=0.5, y=-0.22, xref='paper', yref='paper',
                showarrow=False, font=dict(size=12, color=TEXT_MUTED),
            )],
        )

        # ── 3. SHAP Importance ────────────────────────────────────────────
        X_all_sc  = (X_arr - mean_) / scale_
        shap_vals = X_all_sc * coef           # (N, 11) — LinearExplainer manual
        mean_shap = np.abs(shap_vals).mean(axis=0)

        shap_df = pd.DataFrame({'feature': feat_labels, 'importancia': mean_shap})
        shap_df = shap_df.sort_values('importancia', ascending=True)

        shap_fig = go.Figure(go.Bar(
            x=shap_df['importancia'],
            y=shap_df['feature'],
            orientation='h',
            marker_color=ROXO,
            hovertemplate='<b>%{y}</b><br>|SHAP| médio: %{x:.4f}<extra></extra>',
        ))
        shap_fig.update_layout(
            title='Figura 18 — SHAP: Importância e Impacto das Variáveis',
            xaxis_title='Importância média (|SHAP value|)',
            paper_bgcolor='white', plot_bgcolor='white',
            height=420, font=dict(size=12, family='Segoe UI'),
            title_font_size=13,
            margin=dict(t=60, b=40, l=155, r=40),
        )
        shap_fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')

        # ── 4. Princípio de Pareto ────────────────────────────────────────
        pareto_df = shap_df.sort_values('importancia', ascending=False).copy()
        pareto_df['pct']       = pareto_df['importancia'] / pareto_df['importancia'].sum() * 100
        pareto_df['cumulativo'] = pareto_df['pct'].cumsum()

        pareto_fig = go.Figure()
        pareto_fig.add_trace(go.Bar(
            x=pareto_df['feature'], y=pareto_df['pct'],
            name='Importância %', marker_color='#1976d2',
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
        # Linha dos 80%
        pareto_fig.add_trace(go.Scatter(
            x=pareto_df['feature'].tolist(), y=[80] * len(pareto_df),
            mode='lines', name='80%', yaxis='y2',
            line=dict(color='gray', dash='dash', width=1.5),
            hoverinfo='skip', showlegend=True,
        ))
        pareto_fig.update_layout(
            title='Figura 19 — Princípio de Pareto: Causa-Efeito das Features',
            yaxis=dict(title='Importância individual (%)', showgrid=True, gridcolor='#f0f0f0'),
            yaxis2=dict(title='Cumulativo (%)', overlaying='y', side='right', range=[0, 110]),
            plot_bgcolor='white', paper_bgcolor='white',
            height=440, font=dict(size=11, family='Segoe UI'),
            title_font_size=13,
            legend=dict(orientation='h', y=-0.28, x=0.3),
            margin=dict(t=60, b=90, l=60, r=70),
        )
        pareto_fig.update_xaxes(tickangle=-30)

        return corr_fig, cm_fig, shap_fig, pareto_fig

    except Exception:
        return _placeholder, _placeholder, _placeholder, _placeholder


_CORR_FIG, _CM_FIG, _SHAP_FIG, _PARETO_FIG = _build_static_charts()

_GRAPH_CFG = {'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}


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
        ['C1', 'Endividamento Elevado',               'Loan_Repayment / Income',           '> 10%'],
        ['C2', 'Buffer de Emergência Baixo',          'Disposable_Income / Income',         '< 10%'],
        ['C3', 'Gastos Não Essenciais Altos',         '(Eating_Out + Entertainment) / Income', '> 8,5%'],
        ['C4', 'Potencial de Economia Não Realizado', 'Potential_Savings / Income',         '> 8%'],
    ]

    return html.Div([
        kdd_tag('Processamento'),
        html.H2('Processamento dos Dados', style=H2),
        html.P(
            'Como o dataset não possuía uma variável-alvo pronta, a variável Vulnerable foi construída '
            'a partir de regras de negócio validadas empiricamente. Trata-se de uma variável binária '
            '(0 = seguro, 1 = vulnerável): o cliente é classificado como vulnerável quando satisfaz '
            'simultaneamente duas ou mais das quatro condições de risco definidas abaixo.',
            style=P_STYLE
        ),

        html.Div([
            html.H3('Feature Engineering, Indicadores de Saúde Financeira', style=H3),
            html.P(
                'Antes da classificação, foram criados quatro indicadores derivados da renda para '
                'sustentar a construção do target e capturar padrões de risco financeiro:',
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
                    html.Div('Percentual da renda comprometida com pagamento de empréstimos. '
                             'Valores acima de 10% ativam a condição C1 de risco.',
                             style={'color': TEXT_MUTED, 'fontSize': '13px', 'lineHeight': '1.6'}),
                ], style={'flex': '1', 'padding': '18px', 'backgroundColor': BG_PAGE,
                          'borderRadius': '8px', 'borderLeft': f'3px solid {ROXO}'}),

                html.Div([
                    html.Div('buffer_emergencia', style={
                        'fontFamily': 'monospace', 'fontSize': '13px', 'fontWeight': '700',
                        'color': '#0891b2', 'marginBottom': '8px',
                        'backgroundColor': 'rgba(8,145,178,0.06)', 'display': 'inline-block',
                        'padding': '3px 10px', 'borderRadius': '4px',
                    }),
                    html.Div([html.Code('Disposable_Income / Income', style={'fontSize': '12px', 'color': '#7c3aed'})],
                             style={'marginBottom': '6px'}),
                    html.Div('Percentual da renda que sobra livre após descontar a poupança desejada. '
                             'Buffer abaixo de 10% ativa a condição C2 de risco.',
                             style={'color': TEXT_MUTED, 'fontSize': '13px', 'lineHeight': '1.6'}),
                ], style={'flex': '1', 'padding': '18px', 'backgroundColor': BG_PAGE,
                          'borderRadius': '8px', 'borderLeft': '3px solid #0891b2'}),

                html.Div([
                    html.Div('gastos_nao_essenciais', style={
                        'fontFamily': 'monospace', 'fontSize': '13px', 'fontWeight': '700',
                        'color': ERROR, 'marginBottom': '8px',
                        'backgroundColor': 'rgba(229,75,75,0.06)', 'display': 'inline-block',
                        'padding': '3px 10px', 'borderRadius': '4px',
                    }),
                    html.Div([html.Code('(Eating_Out + Entertainment) / Income', style={'fontSize': '12px', 'color': '#7c3aed'})],
                             style={'marginBottom': '6px'}),
                    html.Div('Proporção da renda gasta em itens não essenciais. '
                             'Acima de 8,5% ativa a condição C3 de risco.',
                             style={'color': TEXT_MUTED, 'fontSize': '13px', 'lineHeight': '1.6'}),
                ], style={'flex': '1', 'padding': '18px', 'backgroundColor': BG_PAGE,
                          'borderRadius': '8px', 'borderLeft': f'3px solid {ERROR}'}),

                html.Div([
                    html.Div('perc_economia_potencial', style={
                        'fontFamily': 'monospace', 'fontSize': '13px', 'fontWeight': '700',
                        'color': SUCCESS, 'marginBottom': '8px',
                        'backgroundColor': 'rgba(46,125,50,0.06)', 'display': 'inline-block',
                        'padding': '3px 10px', 'borderRadius': '4px',
                    }),
                    html.Div([html.Code('Potential_Savings / Income', style={'fontSize': '12px', 'color': '#7c3aed'})],
                             style={'marginBottom': '6px'}),
                    html.Div('Soma dos cortes possíveis em relação à renda total. '
                             'Acima de 8% indica economia não realizada (condição C4).',
                             style={'color': TEXT_MUTED, 'fontSize': '13px', 'lineHeight': '1.6'}),
                ], style={'flex': '1', 'padding': '18px', 'backgroundColor': BG_PAGE,
                          'borderRadius': '8px', 'borderLeft': f'3px solid {SUCCESS}'}),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(240px, 1fr))',
                      'gap': '16px'}),
        ], style=CARD),

        html.Div([
            html.H3('Regras de Negócio para Construção do Target', style=H3),
            html.P('O Risk Score de cada cliente varia de 0 a 4. Se o somatório das condições ativas for ≥ 2, '
                   'a variável Vulnerable recebe valor 1 (alto risco). Caso contrário, valor 0 (seguro).',
                   style={**P_STYLE, 'marginBottom': '20px'}),
            html_table(cond_headers, cond_rows),
            info_box([
                html.Strong('Critério de ativação: '),
                'Risk Score ≥ 2  →  Vulnerable = 1  |  Risk Score < 2  →  Vulnerable = 0'
            ], border_color=ERROR),
        ], style=CARD),

        html.Div([
            html.H3('Distribuição da Base após Criação do Target', style=H3),
            html.P('Após remover os 112 clientes com Desired_Savings = 0 (sem meta de poupança definida), a base de análise passa a ter 19.888 registros.', style={**P_STYLE, 'marginBottom': '16px'}),
            html.Div([
                metric_card('Registros analisados', '19.888', ROXO),
                metric_card('Seguros (classe 0)', '15.357  (~77,2%)', SUCCESS),
                metric_card('Vulneráveis (classe 1)', '4.531  (~22,8%)', ERROR),
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
                'Como a variável-alvo (Vulnerable) foi sinteticamente construída a partir de indicadores '
                'financeiros derivados (como o percentual de empréstimo e o buffer de emergência), incluir '
                'essas mesmas variáveis na matriz de features permitiria que os algoritmos fizessem '
                'engenharia reversa da regra de negócio, memorizando o resultado em vez de aprender '
                'padrões preditivos genuínos.',
                style=P_STYLE
            ),
            info_box([
                html.Strong('Decisão: '), 'As features derivadas utilizadas na construção do target foram '
                'intencionalmente excluídas. A matriz final foi composta pelas ', html.Strong('11 variáveis '
                'numéricas brutas'), ' (gastos absolutos e renda).'
            ], border_color=ERROR),
        ], style=CARD),

        html.Div([
            html.H3('Multicolinearidade, Decisão Consciente', style=H3),
            html.P(
                'Ao usar apenas variáveis brutas, introduz-se multicolinearidade (os gastos absolutos '
                'têm alta correlação com a renda). Essa decisão foi intencional: o objetivo era avaliar '
                'empiricamente como cada arquitetura matemática lida com dados altamente colineares.',
                style=P_STYLE
            ),
            dcc.Graph(id='ml-corr-fig', figure=_CORR_FIG, config=_GRAPH_CFG,
                      style={'marginTop': '16px'}),
        ], style=CARD),

        html.Div([
            html.H3('Padronização, StandardScaler', style=H3),
            html.P(
                'Foi aplicada a padronização via StandardScaler (média 0, desvio padrão 1) com o '
                'objetivo de nivelar matematicamente a base de dados. Sem essa etapa, colunas com '
                'valores numericamente altos (como Income) influenciariam os algoritmos de forma '
                'desproporcional em relação às colunas com valores menores (como Age).',
                style=P_STYLE
            ),
            html.Div([
                html.Span(f, style={
                    'backgroundColor': 'rgba(29,18,82,0.08)', 'color': ROXO,
                    'borderRadius': '6px', 'padding': '7px 16px', 'fontSize': '13px', 'fontWeight': '600',
                    'display': 'inline-block',
                }) for f in FEATURES
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '12px 14px', 'marginTop': '16px'}),
        ], style=CARD),
    ])


def classif_mineracao():
    alg_rows = [
        ['Regressão Logística', 'Linear',   'Interpretável, base referência'],
        ['KNN',                 'Distância','Base/interpretável'],
        ['SVM',                 'Kernel',   'Fronteiras não lineares'],
        ['Random Forest',       'Ensemble', 'Bagging de árvores'],
        ['Gradient Boosting',   'Ensemble', 'Boosting sequencial'],
        ['XGBoost',             'Ensemble', 'Boosting otimizado'],
        ['CatBoost v1',         'Ensemble', 'Hiperparâmetros padrão'],
        ['CatBoost v2',         'Ensemble', 'Hiperparâmetros ajustados'],
    ]

    return html.Div([
        kdd_tag('Mineração'),
        html.H2('Mineração de Dados e Modelagem', style=H2),

        html.Div([
            html.H3('Divisão Estratificada da Base', style=H3),
            html.P(
                'Para garantir que os algoritmos não memorizem os dados de treino, a base foi dividida '
                'de forma estratificada, preservando a proporção original entre classes:',
                style=P_STYLE
            ),
            html.Div([
                metric_card('Treino (80%)', '15.910 registros', ROXO),
                metric_card('Teste  (20%)', '3.978 registros',  '#6366f1'),
            ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px'}),
        ], style=CARD),

        html.Div([
            html.H3('Algoritmos Avaliados, 8 Famílias', style=H3),
            html.P(
                'Foram testadas diferentes famílias de aprendizado de máquina para garantir diversidade '
                'analítica e identificar qual arquitetura melhor suporta o padrão de dados:',
                style=P_STYLE
            ),
            html_table(['Algoritmo', 'Paradigma', 'Justificativa'], alg_rows),
        ], style=CARD),

        html.Div([
            html.H3('Critério de Seleção, Recall da Classe Vulnerável', style={**H3, 'color': ERROR}),
            html.P(
                'O critério decisivo de seleção foi o Recall da classe vulnerável (1). Esta escolha '
                'reflete a realidade do negócio:',
                style=P_STYLE
            ),
            info_box([
                html.Strong('Em análise de risco de crédito, '),
                'o prejuízo de aprovar um mau pagador (falso negativo) é muito superior ao custo de '
                'negar crédito por engano a um cliente seguro (falso positivo). Portanto, maximizar '
                'o Recall da classe 1 é a prioridade.'
            ], border_color=ERROR),
        ], style=CARD),
    ])


def classif_resultados():
    metrics_headers = ['Modelo', 'Métrica', 'Seguro (0)', 'Vulnerável (1)', 'Média', 'Ponderada']
    metrics_rows = [
        ['SVM',                  'Precisão',  '0,971', '0,739', '0,855', '0,918'],
        ['SVM',                  'Recall',    '0,906', '0,907', '0,906', '0,906'],
        ['SVM',                  'F1-Score',  '0,937', '0,815', '0,876', '0,909'],
        ['SVM',                  'Acurácia',  '—',     '—',     '0,906', '—'],
        ['CatBoost v1',          'Precisão',  '0,948', '0,903', '0,925', '0,937'],
        ['CatBoost v1',          'Recall',    '0,974', '0,818', '0,896', '0,938'],
        ['CatBoost v1',          'F1-Score',  '0,961', '0,858', '0,909', '0,937'],
        ['CatBoost v1',          'Acurácia',  '—',     '—',     '0,938', '—'],
        ['CatBoost v2',          'Precisão',  '0,968', '0,827', '0,898', '0,936'],
        ['CatBoost v2',          'Recall',    '0,945', '0,894', '0,920', '0,933'],
        ['CatBoost v2',          'F1-Score',  '0,956', '0,859', '0,908', '0,934'],
        ['CatBoost v2',          'Acurácia',  '—',     '—',     '0,933', '—'],
        ['Regressão Logística ★','Precisão',  '0,980', '0,710', '0,840', '0,910'],
        ['Regressão Logística ★','Recall',    '0,890', '0,920', '0,910', '0,900'],
        ['Regressão Logística ★','F1-Score',  '0,930', '0,800', '0,870', '0,900'],
        ['Regressão Logística ★','Acurácia',  '—',     '—',     '0,900', '—'],
        ['Regressão Logística ★','AUC-ROC',   '—',     '—',     '—',     '0,9506'],
        ['Random Forest',        'Precisão',  '0,940', '0,920', '0,930', '0,940'],
        ['Random Forest',        'Recall',    '0,980', '0,800', '0,890', '0,940'],
        ['Random Forest',        'F1-Score',  '0,960', '0,850', '0,905', '0,940'],
        ['Random Forest',        'Acurácia',  '—',     '—',     '0,940', '—'],
        ['Random Forest',        'AUC-ROC',   '—',     '—',     '—',     '0,9724'],
        ['Gradient Boosting',    'Precisão',  '0,940', '0,880', '0,910', '0,920'],
        ['Gradient Boosting',    'Recall',    '0,970', '0,780', '0,880', '0,930'],
        ['Gradient Boosting',    'F1-Score',  '0,950', '0,830', '0,890', '0,920'],
        ['Gradient Boosting',    'Acurácia',  '—',     '—',     '0,930', '—'],
        ['Gradient Boosting',    'AUC-ROC',   '—',     '—',     '—',     '0,9631'],
        ['XGBoost',              'Precisão',  '0,960', '0,840', '0,900', '0,930'],
        ['XGBoost',              'Recall',    '0,950', '0,870', '0,910', '0,930'],
        ['XGBoost',              'F1-Score',  '0,960', '0,850', '0,900', '0,930'],
        ['XGBoost',              'Acurácia',  '—',     '—',     '0,930', '—'],
        ['XGBoost',              'AUC-ROC',   '—',     '—',     '—',     '0,9759'],
        ['KNN',                  'Precisão',  '0,920', '0,840', '0,880', '0,900'],
        ['KNN',                  'Recall',    '0,960', '0,740', '0,850', '0,910'],
        ['KNN',                  'F1-Score',  '0,940', '0,780', '0,860', '0,900'],
        ['KNN',                  'Acurácia',  '—',     '—',     '0,910', '—'],
        ['KNN',                  'AUC-ROC',   '—',     '—',     '—',     '0,9277'],
    ]

    seg_headers = ['Nível de Risco', 'Clientes', '% da Base', 'Prob. Média', 'Empréstimo Médio', 'Exposição Mensal']
    seg_rows = [
        ['Baixo (< 30%)', '11.629', '58,5%', '7,6%',  'R$ 374',   'R$ 4.350.889'],
        ['Médio (30–60%)', '2.999', '15,1%', '43,3%', 'R$ 1.252', 'R$ 3.754.638'],
        ['Alto (> 60%)',   '5.260', '26,4%', '86,7%', 'R$ 6.073', 'R$ 31.945.316'],
    ]

    return html.Div([
        kdd_tag('Resultados'),
        html.H2('Resultados e Análise do Modelo de Classificação', style=H2),

        # Modelo selecionado
        html.Div([
            html.Div([
                html.Div([badge('MODELO SELECIONADO', SUCCESS)], style={'marginBottom': '8px'}),
                html.H3('Regressão Logística', style={**H3, 'marginTop': '4px', 'fontSize': '22px'}),
                html.P(
                    'Entre os oito algoritmos comparados, a Regressão Logística foi selecionada por '
                    'apresentar o melhor Recall da classe vulnerável (0,92). Embora outros modelos como '
                    'CatBoost e XGBoost apresentem maior acurácia geral, o critério de negócio prioriza '
                    'a detecção correta dos vulneráveis, onde a Regressão Logística se destaca.',
                    style=P_STYLE
                ),
                html.Div([
                    metric_card('Recall (Vulnerável)',  '0,92',   SUCCESS),
                    metric_card('Acurácia Geral',       '90,0%',  ROXO),
                    metric_card('AUC-ROC',              '0,9506', '#6366f1'),
                    metric_card('F1 (Vulnerável)',      '0,800',  '#0891b2'),
                ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(160px, 1fr))',
                          'gap': '16px', 'marginTop': '16px'}),
            ]),
        ], style=CARD),

        # Comparativo completo de algoritmos
        html.Div([
            html.H3('Comparativo Completo, 8 Algoritmos (Tabela 5)', style=H3),
            html.P('Linhas marcadas com ★ indicam o modelo selecionado para produção.', style={**P_STYLE, 'marginBottom': '16px'}),
            html.Div(html_table(metrics_headers, metrics_rows),
                     style={'overflowX': 'auto'}),
        ], style=CARD),

        # Matriz de Confusão
        html.Div([
            html.H3('Interpretabilidade, Matriz de Confusão', style=H3),
            html.P(
                'O modelo alcançou Recall de 0,92, identificando corretamente 837 dos 906 clientes '
                'vulneráveis reais. Os 69 casos restantes (8%) representam falsos negativos. '
                'Na prática, a cada 10 clientes vulneráveis o modelo acerta aproximadamente 9.',
                style=P_STYLE
            ),
            dcc.Graph(id='ml-cm-fig', figure=_CM_FIG, config=_GRAPH_CFG,
                      style={'marginTop': '16px'}),
        ], style=CARD),

        # SHAP e Pareto
        html.Div([
            html.H3('Explicabilidade, SHAP e Princípio de Pareto', style=H3),
            html.P(
                'A interpretabilidade do modelo foi obtida via SHAP, técnica que quantifica a '
                'contribuição de cada variável na previsão individual. A análise de Pareto '
                'confirmou que um conjunto reduzido de variáveis (~27%), com destaque para '
                'endividamento e renda disponível, responde pela maior parte do poder preditivo.',
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
                html.Strong('Variáveis mais impactantes: '),
                'Loan_Repayment e Disposable_Income, ambas interpretáveis e diretamente acionáveis '
                'em estratégias de risco de crédito. Aproximadamente 27% das features respondem '
                'pela maior parte do poder preditivo do modelo.'
            ]),
        ], style=CARD),

        # Segmentação por probabilidade
        html.Div([
            html.H3('Segmentação por Probabilidade (Tabela 6)', style=H3),
            html.P(
                'Em vez da classificação binária, o modelo foi aplicado à base completa via '
                'predict_proba para segmentar clientes em três faixas operacionais de risco. '
                'A lógica: dois clientes classificados como vulneráveis podem exigir respostas '
                'diferentes com base em sua probabilidade individual.',
                style=P_STYLE
            ),
            html_table(seg_headers, seg_rows),
            info_box([
                html.Strong('Conclusão: '),
                '8.259 clientes em risco médio/alto (41,5% da base) representam exposição mensal de '
                'R$ 35,7 milhões, com potencial de R$ 10,7 milhões em inadimplências evitáveis '
                'mediante intervenção antecipada em 30% dos casos.'
            ], border_color=SUCCESS),
        ], style=CARD),

        # ── Previsão Individual ──────────────────────────────────────────────────
        html.Div([
            html.Div([badge('PREVISÃO INDIVIDUAL')], style={'marginBottom': '12px'}),
            html.H3('Simular Vulnerabilidade de um Cliente', style={**H3, 'marginTop': '0'}),
            html.P(
                'Preencha os dados financeiros do cliente e execute o modelo de Regressão Logística '
                'para obter a probabilidade de vulnerabilidade e os indicadores de risco individuais.',
                style={**P_STYLE, 'marginBottom': '24px'}
            ),

            # Grid de inputs
            html.Div([
                # Coluna 1 — Perfil & Renda
                html.Div([
                    html.Div('Perfil & Renda', style={
                        'fontWeight': '700', 'color': ROXO, 'fontSize': '12px',
                        'textTransform': 'uppercase', 'letterSpacing': '0.8px', 'marginBottom': '16px',
                    }),
                    _ip_field('income',       'Renda Mensal (R$)',           35000),
                    _ip_field('age',          'Idade',                       35),
                    _ip_field('dependents',   'Nº de Dependentes',            2),
                    _ip_field('loan',         'Parcela de Empréstimo (R$)',  3000),
                    _ip_field('disposable',   'Renda Disponível (R$)',       8000),
                    _ip_field('savings',      'Poupança Desejada (R$)',      2500),
                ], style={'flex': '1', 'minWidth': '240px'}),

                # Coluna 2 — Gastos Mensais
                html.Div([
                    html.Div('Gastos Mensais', style={
                        'fontWeight': '700', 'color': '#0891b2', 'fontSize': '12px',
                        'textTransform': 'uppercase', 'letterSpacing': '0.8px', 'marginBottom': '16px',
                    }),
                    _ip_field('rent',          'Aluguel (R$)',         7000),
                    _ip_field('groceries',     'Supermercado (R$)',    3000),
                    _ip_field('eating-out',    'Alimentação Fora (R$)',2000),
                    _ip_field('entertainment', 'Entretenimento (R$)',  1500),
                    _ip_field('healthcare',    'Saúde (R$)',            800),
                    _ip_field('pot-savings',   'Potencial de Economia (R$)', 1000,
                              hint='Estime quanto poderia economizar reduzindo gastos supérfluos '
                                   '(ex.: alimentação fora, streaming, entretenimento). '
                                   'Usado para análise da condição C4 — não altera o modelo de IA.'),
                ], style={'flex': '1', 'minWidth': '240px'}),
            ], style={'display': 'flex', 'gap': '40px', 'flexWrap': 'wrap', 'marginBottom': '24px',
                      'backgroundColor': BG_PAGE, 'padding': '24px', 'borderRadius': '10px',
                      'border': f'1px solid {BORDER}'}),

            html.Button('Prever Vulnerabilidade', id='ip-predict-btn', n_clicks=0, style={
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
                'Execute o modelo de vulnerabilidade na base de 20.000 registros e explore a '
                'distribuição de risco com threshold ajustável.',
                style={**P_STYLE, 'marginBottom': '20px'}
            ),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span('Sensibilidade (Threshold):', style={'fontWeight': '600'}),
                        html.Span(' '),
                        html.Span('0.50', id='ml-threshold-val', style={
                            'fontWeight': '700', 'color': ROXO,
                            'backgroundColor': '#f0eeff', 'borderRadius': '4px',
                            'padding': '2px 8px', 'fontSize': '13px',
                        }),
                    ], style={'marginBottom': '10px'}),
                    dcc.Slider(
                        id='ml-threshold', min=0.0, max=1.0, step=0.01, value=0.5,
                        marks={
                            0.0: {'label': 'Mais Seguros',  'style': {'whiteSpace': 'nowrap', 'color': SUCCESS}},
                            0.5: {'label': 'Padrão',        'style': {'whiteSpace': 'nowrap', 'color': TEXT_MUTED}},
                            1.0: {'label': 'Mais Críticos', 'style': {'whiteSpace': 'nowrap', 'color': ERROR}},
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
            section_img(
                '/assets/pca.png',
                'Gráfico 27, Visualização 2D dos clusters K-Means++ por projeção PCA.',
                max_width='680px'
            ),
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
            section_img(
                '/assets/correlacao-cluster.png',
                'Gráfico 4, Matriz de correlação das features transformadas (clustering).',
                max_width='680px'
            ),
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
            section_img(
                '/assets/epsilon.png',
                'Gráfico 5, K-Distance para determinação do epsilon ideal.',
                max_width='620px'
            ),
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
            section_img(
                '/assets/silhouette-cotovelo.png',
                'Gráfico 6, Método do Cotovelo e Silhouette Score para seleção de K.',
                max_width='700px'
            ),
            html.H4('Silhouette Diagram por Cluster (Gráfico 21)', style={'color': ROXO, 'marginTop': '20px', 'marginBottom': '10px'}),
            html.P(
                'O Silhouette Diagram detalha a largura de silhouette individual de cada amostra '
                'agrupada por cluster, permitindo identificar grupos subótimos e confirmar que '
                'K=5 produz separação homogênea sem clusters dominantes ou degenerados.',
                style=P_STYLE
            ),
            section_img(
                '/assets/silhouette-diagram-correto.png',
                'Gráfico 21, Silhouette Diagram por cluster (K-Means++).',
                max_width='700px'
            ),
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
            section_img(
                '/assets/visualizacao-2d-correto.png',
                'Gráfico 27, Visualização 2D dos clusters K-Means++ por projeção PCA.',
                max_width='680px'
            ),

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
                    section_img('/assets/cluster_shap.png', 'Gráfico 30, SHAP Global do modelo K-Means++.', max_width='100%'),
                ], style={'flex': '1', 'minWidth': '280px'}),

                html.Div([
                    html.H4('Princípio de Pareto, Comprometimento por Cluster (Gráfico 29)', style={'color': ROXO, 'marginBottom': '10px', 'fontSize': '15px'}),
                    html.P(
                        'Rent_pct é a categoria de maior peso em quatro dos cinco clusters. '
                        'A exceção é o Cluster 3 (O Poupador Agressivo), onde Loan_Repayment_pct assume '
                        'a posição dominante, reforçando a separação comportamental entre os perfis.',
                        style={**P_STYLE, 'marginBottom': '12px'}
                    ),
                    section_img('/assets/cluster_pareto.png', 'Gráfico 29, Princípio de Pareto por cluster (K-Means++).', max_width='100%'),
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
                'Em combinação com o modelo de classificação, é possível não apenas prever '
                'o risco de inadimplência, mas identificar o padrão financeiro que origina esse risco.',
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
                    html.Div('Determina se há probabilidade de inadimplência para aquele perfil.', style={'color': TEXT_MUTED, 'fontSize': '14px'}),
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
        Retorna: (coef, intercept, scaler_mean, scaler_scale, features)
        """
        p = joblib.load(MODEL_PATH)
        coef   = np.array(p['coef'])          # (1, 11)
        intercept = np.array(p['intercept'])  # (1,)
        mean_  = np.array(p['scaler_mean'])   # (11,)
        scale_ = np.array(p['scaler_scale'])  # (11,)
        return coef, intercept, mean_, scale_, p['features']

    def _scale(X_arr, mean_, scale_):
        """StandardScaler puro numpy."""
        return (X_arr - mean_) / scale_

    def _predict_proba(X_scaled, coef, intercept):
        """Regressão Logística — sigmoid puro numpy."""
        z = X_scaled @ coef.T + intercept   # (n, 1)
        return 1.0 / (1.0 + np.exp(-z))     # (n, 1)

    def _load_base():
        """Carrega dataset, filtra Desired_Savings > 0 → 19.888 registros.
        Tenta parquet primeiro; cai para CSV se pyarrow não estiver instalado."""
        try:
            df = pd.read_parquet(DATA_PATH)
        except Exception:
            csv_path = DATA_PATH.replace('.parquet', '.csv')
            df = pd.read_csv(csv_path)
        df = df[df['Desired_Savings'] > 0].reset_index(drop=True)
        return df

    def _nivel(p):
        if p < 0.30: return 'Baixo'
        if p < 0.60: return 'Médio'
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
            coef, intercept, mean_, scale_, features = _load_model()
            df = _load_base()

            faltando = [c for c in features if c not in df.columns]
            if faltando:
                empty = go.Figure()
                err = html.Div(f'Colunas ausentes: {", ".join(faltando)}', style={'color': ERROR})
                return empty, empty, err, {}

            X       = df[features].fillna(0).to_numpy(dtype=float)
            X_sc    = _scale(X, mean_, scale_)
            probs   = _predict_proba(X_sc, coef, intercept).ravel()
            df['prob_risco'] = probs

        except Exception as exc:
            empty = go.Figure()
            err = html.Div(f'Erro ao carregar o modelo: {exc}', style={'color': ERROR, 'fontSize': '14px'})
            return empty, empty, err, {}

        df['classif'] = (df['prob_risco'] >= threshold).astype(int)
        df['nivel']   = df['prob_risco'].apply(_nivel)

        # Gráfico de pizza — proporção Seguro × Vulnerável com o threshold atual
        classif_labels = df['classif'].map({0: 'Seguro', 1: 'Vulnerável'})
        pie = px.pie(
            classif_labels.rename('Classificação').to_frame(),
            names='Classificação',
            title=f'Seguro × Vulnerável (threshold = {threshold:.2f})',
            color='Classificação',
            color_discrete_map={'Seguro': SUCCESS, 'Vulnerável': ERROR},
            hole=0.4,
        )
        pie.update_traces(textposition='inside', textinfo='percent+label')
        pie.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          showlegend=False, font_family='Segoe UI',
                          margin=dict(t=50, b=20, l=20, r=20))

        # Histograma
        hist = px.histogram(df, x='prob_risco', nbins=50,
                            title='Distribuição de Probabilidades de Risco',
                            labels={'prob_risco': 'Probabilidade de Vulnerabilidade'},
                            color_discrete_sequence=['#594CA3'])
        hist.add_vline(x=threshold, line_dash='dash', line_color=ERROR,
                       annotation_text=f'Threshold ({threshold:.2f})',
                       annotation_position='top right')
        hist.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                           yaxis_title='Nº de Clientes', font_family='Segoe UI',
                           margin=dict(t=50, b=40, l=40, r=20))
        hist.update_xaxes(showgrid=False)
        hist.update_yaxes(gridcolor='#f0f0f0')

        total    = len(df)
        seguros  = (df['classif'] == 0).sum()
        vuln     = (df['classif'] == 1).sum()
        alto_df  = df[df['nivel'] == 'Alto']
        exp_alto = alto_df['Loan_Repayment'].sum() if 'Loan_Repayment' in alto_df.columns else 0

        cards = html.Div([
            _result_card('Registros Analisados', f'{total:,}',       ROXO),
            _result_card('Seguros',              f'{seguros:,}',      SUCCESS),
            _result_card('Vulneráveis',          f'{vuln:,}',         ERROR),
            _result_card('Exposição Alto Risco', f'R$ {exp_alto:,.0f}', '#f59e0b'),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(180px, 1fr))',
                  'gap': '14px', 'marginBottom': '20px'})

        return (
            pie, hist, cards,
            df[['prob_risco', 'classif', 'nivel']].to_json(orient='records', force_ascii=False),
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
            'Dependents':        field_map.get('dependents', 0),
            'Loan_Repayment':    field_map.get('loan', 0),
            'Eating_Out':        field_map.get('eating-out', 0),
            'Entertainment':     field_map.get('entertainment', 0),
            'Healthcare':        field_map.get('healthcare', 0),
            'Rent':              field_map.get('rent', 0),
            'Groceries':         field_map.get('groceries', 0),
            'Disposable_Income': field_map.get('disposable', 0),
            'Desired_Savings':   field_map.get('savings', 0),
        }
        pot_savings = field_map.get('pot-savings', 0)

        if vals['Income'] <= 0:
            return html.Div('⚠️ Renda deve ser maior que zero.', style={'color': ERROR, 'fontWeight': '600'})

        try:
            coef, intercept, mean_, scale_, features = _load_model()
        except Exception as exc:
            return html.Div(f'Erro ao carregar modelo: {exc}', style={'color': ERROR})

        X_row  = np.array([[vals[f] for f in features]], dtype=float)
        X_sc   = _scale(X_row, mean_, scale_)
        prob   = float(_predict_proba(X_sc, coef, intercept)[0, 0])
        nivel = _nivel(prob)

        # Indicadores derivados
        inc       = vals['Income']
        perc_emp  = vals['Loan_Repayment'] / inc
        perc_ness = (vals['Eating_Out'] + vals['Entertainment']) / inc
        buffer    = (vals['Disposable_Income'] - vals['Desired_Savings']) / inc
        perc_pot  = pot_savings / inc if inc > 0 else 0

        conds = [
            ('C1 — Endividamento elevado',        perc_emp  > 0.10,  f'{perc_emp*100:.1f}%',   'Empréstimo > 10% da renda'),
            ('C2 — Buffer de emergência baixo',   buffer    < 0.10,  f'{buffer*100:.1f}%',     'Buffer < 10% da renda'),
            ('C3 — Gastos não essenciais altos',  perc_ness > 0.085, f'{perc_ness*100:.1f}%',  'Não essenciais > 8,5% da renda'),
            ('C4 — Potencial de economia alto',   perc_pot  > 0.08,  f'R$ {pot_savings:,.0f}', 'Potencial > 8% da renda'),
        ]
        risk_count = sum(1 for _, ativo, *_ in conds if ativo)

        # Gauge de probabilidade
        gauge_color = ERROR if nivel == 'Alto' else ('#f59e0b' if nivel == 'Médio' else SUCCESS)
        gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=round(prob * 100, 1),
            number={'suffix': '%', 'font': {'size': 32, 'color': gauge_color}},
            gauge={
                'axis': {'range': [0, 100], 'ticksuffix': '%'},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 30],   'color': 'rgba(46,125,50,0.15)'},
                    {'range': [30, 60],  'color': 'rgba(245,158,11,0.15)'},
                    {'range': [60, 100], 'color': 'rgba(229,75,75,0.15)'},
                ],
                'threshold': {'line': {'color': gauge_color, 'width': 3}, 'value': prob * 100},
            },
            title={'text': 'Probabilidade de Vulnerabilidade', 'font': {'size': 14, 'color': TEXT_MUTED}},
        ))
        gauge.update_layout(
            height=260, paper_bgcolor='white', font_family='Segoe UI',
            margin=dict(t=40, b=10, l=20, r=20),
        )

        # Badge de nível
        badge_colors = {'Baixo': SUCCESS, 'Médio': '#f59e0b', 'Alto': ERROR}
        badge_txt    = {'Baixo': 'RISCO BAIXO', 'Médio': 'RISCO MÉDIO', 'Alto': 'RISCO ALTO'}

        nivel_badge = html.Span(badge_txt[nivel], style={
            'backgroundColor': badge_colors[nivel], 'color': BRANCO,
            'borderRadius': '6px', 'padding': '6px 18px',
            'fontSize': '13px', 'fontWeight': '800', 'letterSpacing': '1px',
            'display': 'inline-block',
        })

        # Linha de condições ativas
        cond_items = []
        for nome, ativo, valor, descr in conds:
            cor   = ERROR if ativo else SUCCESS
            icone = '✗' if ativo else '✓'
            cond_items.append(html.Div([
                html.Div([
                    html.Span(icone, style={'color': cor, 'fontWeight': '800', 'marginRight': '8px', 'fontSize': '16px'}),
                    html.Span(nome, style={'fontWeight': '600', 'fontSize': '13px', 'color': TEXT_MAIN}),
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px'}),
                html.Div([
                    html.Span('Valor: ', style={'color': TEXT_MUTED, 'fontSize': '12px'}),
                    html.Span(valor, style={'fontWeight': '700', 'color': cor, 'fontSize': '12px'}),
                    html.Span(f'  ·  {descr}', style={'color': TEXT_MUTED, 'fontSize': '12px'}),
                ]),
            ], style={
                'padding': '12px 16px', 'borderRadius': '8px', 'marginBottom': '8px',
                'backgroundColor': f'rgba(229,75,75,0.05)' if ativo else f'rgba(46,125,50,0.05)',
                'borderLeft': f'3px solid {cor}',
            }))

        # ── Classificação binária pelas condições (risk_score >= 2) ─────────────
        is_vulnerable  = risk_count >= 2
        bin_color      = ERROR if is_vulnerable else SUCCESS
        bin_label      = 'VULNERÁVEL' if is_vulnerable else 'SEGURO'
        bin_descr      = (
            f'{risk_count} de 4 condições de risco ativas — score ≥ 2 → Vulnerável.'
            if is_vulnerable else
            f'{risk_count} de 4 condições de risco ativas — score < 2 → Seguro.'
        )
        binary_badge = html.Span(bin_label, style={
            'backgroundColor': bin_color, 'color': BRANCO,
            'borderRadius': '6px', 'padding': '6px 18px',
            'fontSize': '13px', 'fontWeight': '800', 'letterSpacing': '1px',
            'display': 'inline-block',
        })

        # ── Textos explicativos ───────────────────────────────────────────────
        nivel_desc = {
            'Alto':  f'O modelo detectou padrão de vulnerabilidade elevado na combinação das 11 variáveis. (< 30% = Baixo · 30–60% = Médio · > 60% = Alto)',
            'Médio': f'Indicadores com pontos de atenção. Monitoramento preventivo indicado. (< 30% = Baixo · 30–60% = Médio · > 60% = Alto)',
            'Baixo': f'Baixa propensão a vulnerabilidade com base nas 11 variáveis analisadas. (< 30% = Baixo · 30–60% = Médio · > 60% = Alto)',
        }

        _sec_label = lambda txt: html.Div(txt, style={
            'fontSize': '11px', 'color': TEXT_MUTED, 'fontWeight': '700',
            'textTransform': 'uppercase', 'letterSpacing': '0.8px', 'marginBottom': '12px',
        })

        return html.Div([
            # ── Painel superior: dois resultados lado a lado ──────────────────
            html.Div([

                # Bloco 1 — Regressão Logística (probabilidade)
                html.Div([
                    _sec_label('Predição do Modelo — Regressão Logística'),
                    html.Div(dcc.Graph(figure=gauge, config={'displayModeBar': False})),
                    html.Div(nivel_badge, style={'textAlign': 'center', 'marginTop': '8px'}),
                    html.P(f'{prob*100:.1f}% de probabilidade de vulnerabilidade. {nivel_desc[nivel]}',
                           style={'color': TEXT_MUTED, 'fontSize': '12px', 'lineHeight': '1.6',
                                  'marginTop': '10px', 'marginBottom': '0'}),
                ], style={
                    'flex': '1', 'minWidth': '280px',
                    'backgroundColor': BG_PAGE, 'padding': '20px', 'borderRadius': '10px',
                    'border': f'1px solid {BORDER}',
                }),

                # Bloco 2 — Condições (classificação binária)
                html.Div([
                    _sec_label('Classificação por Condições de Risco (C1–C4)'),
                    html.Div(binary_badge, style={'marginBottom': '10px'}),
                    html.P(bin_descr, style={
                        'color': TEXT_MUTED, 'fontSize': '13px', 'lineHeight': '1.6',
                        'marginBottom': '16px',
                    }),
                    html.Div(
                        'Score = nº de condições ativas. Score ≥ 2 → Vulnerável. '
                        'Critério usado para definir o target de treinamento do modelo.',
                        style={'fontSize': '11px', 'color': TEXT_MUTED, 'fontStyle': 'italic',
                               'marginBottom': '16px'},
                    ),
                    *cond_items,
                ], style={
                    'flex': '1', 'minWidth': '280px',
                    'backgroundColor': BG_PAGE, 'padding': '20px', 'borderRadius': '10px',
                    'border': f'1px solid {BORDER}',
                }),

            ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}),
        ])
