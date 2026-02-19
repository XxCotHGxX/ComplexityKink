"""
Modern Plotly-based visualizations for the Complexity Kink paper.
Produces interactive HTML + static PNG exports.
"""
import json
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ── Data Loading ──────────────────────────────────────────────────────
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, 'data', 'iv_enriched_dataset.jsonl')
    model_path = os.path.join(base, 'output', 'kappa_predictor_stage1.joblib')

    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            row = dict(item.get('iv_features', {}))
            row['kappa_actual'] = item.get('kappa_cyclomatic', 1)
            status = item.get('status', [])
            row['pass_rate'] = sum(1 for s in status if 'pass' in s.lower()) / len(status) if status else 0
            row['is_success'] = item.get('is_success', 0)
            row['e_norm'] = item.get('e_norm', 0)
            data.append(row)

    df = pd.DataFrame(data)
    model = joblib.load(model_path)
    feat_cols = ['inst_tokens','inst_if_count','inst_loop_count','inst_class_count',
                 'inst_func_count','inst_logic_count','inst_total_structural','inst_avg_word_len']
    df['kappa_predicted'] = model.predict(df[feat_cols].values)
    return df

# ── Color Palette ─────────────────────────────────────────────────────
COLORS = {
    'bg': '#0d1117',
    'card': '#161b22',
    'text': '#e6edf3',
    'muted': '#7d8590',
    'accent': '#58a6ff',
    'green': '#3fb950',
    'red': '#f85149',
    'orange': '#d29922',
    'purple': '#bc8cff',
    'grid': '#21262d',
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS['bg'],
    plot_bgcolor=COLORS['card'],
    font=dict(family='Inter, sans-serif', color=COLORS['text'], size=13),
    margin=dict(l=60, r=30, t=80, b=60),
)

def style_axes(fig):
    fig.update_xaxes(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], linecolor=COLORS['grid'])
    fig.update_yaxes(gridcolor=COLORS['grid'], zerolinecolor=COLORS['grid'], linecolor=COLORS['grid'])
    return fig

# ── Figure 1: Imposter Violin Plot ───────────────────────────────────
def fig_imposter_violins(df, outdir):
    """Show predicted kappa distributions per observed kappa bin -- exposes imposters."""
    subset = df[df['kappa_actual'].between(1, 8)].copy()
    subset['kappa_obs_label'] = 'κ_obs = ' + subset['kappa_actual'].astype(str)

    fig = go.Figure()

    for k in range(1, 9):
        bin_data = subset[subset['kappa_actual'] == k]['kappa_predicted']
        color = COLORS['red'] if k == 1 else COLORS['accent']
        fig.add_trace(go.Violin(
            y=bin_data,
            name=f'κ={k}',
            box_visible=True,
            meanline_visible=True,
            fillcolor=color if k == 1 else None,
            opacity=0.85 if k == 1 else 0.6,
            line_color=color,
            marker=dict(color=color, size=2),
        ))

    fig.add_hline(y=6.5, line_dash='dash', line_color=COLORS['orange'],
                  annotation_text='Complexity Kink (κ=6.5)',
                  annotation_position='top right',
                  annotation_font_color=COLORS['orange'])

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text='Where Do Tasks Really Belong?<br><sup style="color:#7d8590">Predicted complexity distribution per observed complexity bin</sup>',
                   font_size=18),
        xaxis_title='Observed Complexity Bin',
        yaxis_title='Predicted Target Complexity (κ̂)',
        showlegend=False,
        height=550,
        width=900,
    )
    style_axes(fig)

    fig.write_html(os.path.join(outdir, 'imposter_violins.html'))
    fig.write_image(os.path.join(outdir, 'imposter_violins.png'), scale=2)
    print(f'  Saved imposter_violins')

# ── Figure 2: Sankey Flow Diagram ────────────────────────────────────
def fig_sankey_flow(df, outdir):
    """Shows how tasks flow from predicted bins to observed bins (misclassification)."""
    df_copy = df.copy()
    df_copy['pred_bin'] = pd.cut(df_copy['kappa_predicted'],
                                  bins=[0, 3, 6.5, 10, 15, 50],
                                  labels=['κ̂ 0-3', 'κ̂ 3-6.5', 'κ̂ 6.5-10', 'κ̂ 10-15', 'κ̂ 15+'])
    df_copy['obs_bin'] = pd.cut(df_copy['kappa_actual'],
                                 bins=[0, 1, 3, 6.5, 10, 50],
                                 labels=['κ=1', 'κ 2-3', 'κ 4-6', 'κ 7-10', 'κ 11+'])

    flow = df_copy.groupby(['pred_bin', 'obs_bin']).size().reset_index(name='count')
    flow = flow[flow['count'] > 50]  # filter noise

    pred_labels = list(flow['pred_bin'].unique())
    obs_labels = list(flow['obs_bin'].unique())
    all_labels = pred_labels + obs_labels

    source_idx = [pred_labels.index(r) for r in flow['pred_bin']]
    target_idx = [len(pred_labels) + obs_labels.index(r) for r in flow['obs_bin']]

    # Color: red for flows that go to wrong bin, blue for correct
    link_colors = []
    for _, row in flow.iterrows():
        pred = str(row['pred_bin'])
        obs = str(row['obs_bin'])
        # Misclassified = predicted high but observed at κ=1
        if obs == 'κ=1' and pred not in ['κ̂ 0-3']:
            link_colors.append('rgba(248, 81, 73, 0.4)')  # red
        else:
            link_colors.append('rgba(88, 166, 255, 0.25)')  # blue

    node_colors = [COLORS['purple']] * len(pred_labels) + [COLORS['accent']] * len(obs_labels)

    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color=COLORS['grid'], width=1),
            label=all_labels,
            color=node_colors,
            customdata=[f'{l}' for l in all_labels],
            hovertemplate='%{customdata}: %{value} tasks<extra></extra>',
        ),
        link=dict(
            source=source_idx,
            target=target_idx,
            value=flow['count'].tolist(),
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}: %{value} tasks<extra></extra>',
        ),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text='Task Misclassification Flow<br><sup style="color:#7d8590">Predicted complexity (left) → Observed output complexity (right). Red = misclassified to κ=1</sup>',
                   font_size=18),
        height=500,
        width=900,
    )

    fig.write_html(os.path.join(outdir, 'sankey_misclassification.html'))
    fig.write_image(os.path.join(outdir, 'sankey_misclassification.png'), scale=2)
    print(f'  Saved sankey_misclassification')

# ── Figure 3: The Kink (Modern Scatter + Trend) ──────────────────────
def fig_kink_scatter(df, outdir):
    """Modern scatter plot showing pass rate vs predicted complexity with the kink."""
    # Bin for smoothing
    df_copy = df.copy()
    df_copy['kappa_bin'] = pd.cut(df_copy['kappa_predicted'], bins=40)
    binned = df_copy.groupby('kappa_bin', observed=True).agg(
        mean_kappa=('kappa_predicted', 'mean'),
        mean_pass=('pass_rate', 'mean'),
        count=('pass_rate', 'count'),
        se=('pass_rate', 'sem'),
    ).dropna().reset_index()

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(binned['mean_kappa']) + list(binned['mean_kappa'][::-1]),
        y=list(binned['mean_pass'] + 1.96 * binned['se']) + list((binned['mean_pass'] - 1.96 * binned['se'])[::-1]),
        fill='toself',
        fillcolor='rgba(88, 166, 255, 0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% CI',
        showlegend=True,
        hoverinfo='skip',
    ))

    # Mean trend line
    fig.add_trace(go.Scatter(
        x=binned['mean_kappa'],
        y=binned['mean_pass'],
        mode='lines+markers',
        line=dict(color=COLORS['accent'], width=3),
        marker=dict(size=6, color=COLORS['accent']),
        name='Mean Pass Rate',
        hovertemplate='κ̂ = %{x:.1f}<br>Pass Rate = %{y:.1%}<br><extra></extra>',
    ))

    # Kink line
    fig.add_vline(x=6.5, line_dash='dash', line_color=COLORS['red'], line_width=2)
    fig.add_annotation(x=6.5, y=0.52, text='Complexity Kink<br>κ̂ = 6.5',
                       showarrow=True, arrowhead=2, arrowcolor=COLORS['red'],
                       font=dict(color=COLORS['red'], size=14),
                       ax=60, ay=-40)

    # Regime annotations
    fig.add_annotation(x=3.5, y=0.48, text='<b>Reliable Zone</b><br>Pass Rate ≈ 40%',
                       showarrow=False, font=dict(color=COLORS['green'], size=13))
    fig.add_annotation(x=12, y=0.20, text='<b>Collapse Zone</b><br>Pass Rate ≈ 12%',
                       showarrow=False, font=dict(color=COLORS['red'], size=13))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text='The Complexity Kink<br><sup style="color:#7d8590">LLM pass rate vs predicted target complexity (2SLS-corrected)</sup>',
                   font_size=18),
        xaxis_title='Predicted Target Complexity (κ̂)',
        yaxis_title='Pass Rate',
        yaxis_tickformat='.0%',
        height=500,
        width=900,
        legend=dict(x=0.75, y=0.95, bgcolor='rgba(0,0,0,0)'),
    )
    style_axes(fig)

    fig.write_html(os.path.join(outdir, 'complexity_kink.html'))
    fig.write_image(os.path.join(outdir, 'complexity_kink.png'), scale=2)
    print(f'  Saved complexity_kink')

# ── Figure 4: Phase Heatmap ──────────────────────────────────────────
def fig_phase_heatmap(df, outdir):
    """Interactive heatmap: complexity x entropy → pass rate."""
    df_copy = df.copy()
    df_copy['K_bin'] = pd.cut(df_copy['kappa_predicted'], bins=12,
                               labels=[f'{x:.0f}' for x in np.linspace(2, 16, 12)])
    df_copy['E_bin'] = pd.cut(df_copy['e_norm'], bins=8,
                               labels=[f'{x:.2f}' for x in np.linspace(0.1, 0.9, 8)])

    heatmap = df_copy.pivot_table(index='K_bin', columns='E_bin',
                                   values='pass_rate', aggfunc='mean',
                                   observed=False)

    fig = go.Figure(go.Heatmap(
        z=heatmap.values,
        x=heatmap.columns.astype(str),
        y=heatmap.index.astype(str),
        colorscale=[
            [0.0, '#1a1e2e'],
            [0.15, '#f85149'],
            [0.35, '#d29922'],
            [0.6, '#58a6ff'],
            [1.0, '#3fb950'],
        ],
        colorbar=dict(title='Pass Rate', tickformat='.0%'),
        hovertemplate='Complexity: %{y}<br>Entropy: %{x}<br>Pass Rate: %{z:.1%}<extra></extra>',
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text='Performance Phase Diagram<br><sup style="color:#7d8590">Pass rate by predicted complexity × instruction entropy</sup>',
                   font_size=18),
        xaxis_title='Instruction Entropy (E_norm)',
        yaxis_title='Predicted Complexity (κ̂)',
        height=550,
        width=900,
    )

    fig.write_html(os.path.join(outdir, 'phase_heatmap.html'))
    fig.write_image(os.path.join(outdir, 'phase_heatmap.png'), scale=2)
    print(f'  Saved phase_heatmap')

# ── Figure 5: Before/After Comparison ────────────────────────────────
def fig_before_after(df, outdir):
    """Side-by-side bar chart: naive vs corrected view."""
    # Naive: group by observed kappa
    naive = df[df['kappa_actual'].between(1, 12)].groupby('kappa_actual')['pass_rate'].mean()

    # Corrected: group by predicted kappa (binned to integers)
    df_copy = df.copy()
    df_copy['pred_int'] = df_copy['kappa_predicted'].round().astype(int).clip(1, 12)
    corrected = df_copy.groupby('pred_int')['pass_rate'].mean()

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        '<span style="color:#f85149">Naive View (Output κ) — BIASED</span>',
        '<span style="color:#3fb950">Corrected View (Predicted κ̂) — 2SLS</span>',
    ], horizontal_spacing=0.12)

    fig.add_trace(go.Bar(
        x=naive.index, y=naive.values,
        marker_color=[COLORS['red'] if k == 1 else COLORS['muted'] for k in naive.index],
        hovertemplate='κ_obs=%{x}<br>Pass Rate=%{y:.1%}<extra></extra>',
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=corrected.index, y=corrected.values,
        marker_color=[COLORS['red'] if k > 6.5 else COLORS['green'] for k in corrected.index],
        hovertemplate='κ̂=%{x}<br>Pass Rate=%{y:.1%}<extra></extra>',
        showlegend=False,
    ), row=1, col=2)

    # Kink line on corrected
    fig.add_vline(x=6.5, line_dash='dash', line_color=COLORS['orange'],
                  line_width=2, row=1, col=2)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text='The Reverse Threshold Problem<br><sup style="color:#7d8590">Why output-based metrics are misleading</sup>',
                   font_size=18),
        height=450,
        width=1000,
    )
    fig.update_yaxes(title_text='Pass Rate', tickformat='.0%', row=1, col=1)
    fig.update_yaxes(tickformat='.0%', row=1, col=2)
    fig.update_xaxes(title_text='Observed κ', row=1, col=1)
    fig.update_xaxes(title_text='Predicted κ̂', row=1, col=2)
    style_axes(fig)

    fig.write_html(os.path.join(outdir, 'before_after.html'))
    fig.write_image(os.path.join(outdir, 'before_after.png'), scale=2)
    print(f'  Saved before_after')


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(base, 'output', 'plotly')
    os.makedirs(outdir, exist_ok=True)

    print('Loading data...')
    df = load_data()
    print(f'Loaded {len(df)} samples\n')

    print('Generating visualizations:')
    fig_imposter_violins(df, outdir)
    fig_sankey_flow(df, outdir)
    fig_kink_scatter(df, outdir)
    fig_phase_heatmap(df, outdir)
    fig_before_after(df, outdir)

    print(f'\nAll visualizations saved to {outdir}/')
    print('Open the .html files in a browser for interactive versions!')
