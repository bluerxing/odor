#!/usr/bin/env python3
"""
Generate Figure 3: Composite Empirical Dashboard
=================================================
Re-renders all sub-panels from raw viz_data JSON into a single
publication-quality composite figure (NOT image stitching).

Usage:
    python generate_fig3_composite.py [--json PATH] [--rules PATH]

Defaults:
    --json  ../03_rule_extraction/odor_level_patterns_weighted.json
    --rules ../03_rule_extraction/complex_fol_rules.json
"""

import json
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter, defaultdict
from pathlib import Path

# ─── Global style ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.unicode_minus': False,
    'axes.linewidth': 0.8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,          # we'll add grids selectively
    'axes.spines.top': False,
    'axes.spines.right': False,
})

PANEL_LABEL_KW = dict(fontsize=13, fontweight='bold', va='top', ha='left')

# ─── EC helpers ─────────────────────────────────────────────────────
EC_CATEGORIES = {
    '1': 'Oxidoreductase', '2': 'Transferase', '3': 'Hydrolase',
    '4': 'Lyase',          '5': 'Isomerase',   '6': 'Ligase',
}
EC_COLORS = {
    'Oxidoreductase': '#E74C3C', 'Transferase': '#27AE60',
    'Hydrolase':      '#3498DB', 'Lyase':       '#F39C12',
    'Isomerase':      '#9B59B6', 'Ligase':      '#1ABC9C',
    'Unknown':        '#95A5A6',
}

def ec_category(ec_number: str) -> str:
    first = ec_number.split('.')[0] if ec_number else ''
    return EC_CATEGORIES.get(first, 'Unknown')


# ═══════════════════════════════════════════════════════════════════
#  PANEL RENDERERS — each draws into a given Axes from raw data
# ═══════════════════════════════════════════════════════════════════

def panel_A_odor_distribution(ax_left, ax_right, data):
    """(A) Source vs Target odor distribution — twin horizontal bars."""
    N = 12

    # Source
    src = data.get('top_source_odors', [])[:N]
    s_names = [d['odor'] for d in src]
    s_vals  = [d['total_weight'] for d in src]
    ax_left.barh(range(len(s_names)), s_vals, color='#3498DB', edgecolor='white', linewidth=0.4)
    ax_left.set_yticks(range(len(s_names)))
    ax_left.set_yticklabels(s_names, fontsize=7)
    ax_left.invert_yaxis()
    ax_left.set_xlabel('Weighted freq.', fontsize=7.5)
    ax_left.set_title('Source odors', fontsize=9, fontweight='bold', pad=4)
    ax_left.grid(axis='x', alpha=0.25, linewidth=0.5)

    # Target
    tgt = data.get('top_target_odors', [])[:N]
    t_names = [d['odor'] for d in tgt]
    t_vals  = [d['total_weight'] for d in tgt]
    ax_right.barh(range(len(t_names)), t_vals, color='#E74C3C', edgecolor='white', linewidth=0.4)
    ax_right.set_yticks(range(len(t_names)))
    ax_right.set_yticklabels(t_names, fontsize=7)
    ax_right.invert_yaxis()
    ax_right.set_xlabel('Weighted freq.', fontsize=7.5)
    ax_right.set_title('Target odors', fontsize=9, fontweight='bold', pad=4)
    ax_right.grid(axis='x', alpha=0.25, linewidth=0.5)


def panel_B_pathway_length(ax, data):
    """(B) EC sequence length distribution."""
    triplets = data.get('top_triplets', [])
    sample_size = min(10000, len(triplets))
    sampled = random.sample(triplets, sample_size) if len(triplets) > sample_size else triplets
    lengths = [len(t.get('ec_sequence', [])) for t in sampled]

    if not lengths:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    counts = Counter(lengths)
    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]

    bars = ax.bar(xs, ys, color='#27AE60', edgecolor='white', linewidth=0.5, width=0.7)
    for bar, v in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(ys)*0.02,
                str(v), ha='center', va='bottom', fontsize=6, fontweight='bold')

    avg_l = np.mean(lengths)
    ax.axvline(avg_l, color='#C0392B', ls='--', lw=1, alpha=0.7)
    ax.text(avg_l + 0.1, max(ys)*0.92, f'mean={avg_l:.1f}', fontsize=6.5,
            color='#C0392B', fontweight='bold')

    ax.set_xticks(xs)
    ax.set_xlabel('Pathway length (# EC steps)', fontsize=7.5)
    ax.set_ylabel('Count', fontsize=7.5)
    ax.set_title('Pathway length distribution', fontsize=9, fontweight='bold', pad=4)
    ax.grid(axis='y', alpha=0.2, linewidth=0.5)


def panel_C_heatmap(ax, data):
    """(C) Source × Target odor transformation heatmap."""
    N = 12
    top_s = [d['odor'] for d in data.get('top_source_odors', [])[:N]]
    top_t = [d['odor'] for d in data.get('top_target_odors', [])[:N]]

    matrix = np.zeros((len(top_s), len(top_t)))
    triplets = data.get('top_triplets', [])[:5000]

    for tr in triplets:
        s, t = tr.get('source_odor', ''), tr.get('target_odor', '')
        freq = float(tr.get('weighted_frequency', 0))
        if s in top_s and t in top_t:
            matrix[top_s.index(s), top_t.index(t)] += freq

    # Log-scale for better contrast
    matrix_log = np.log1p(matrix)

    cmap = LinearSegmentedColormap.from_list('custom',
        ['#FDFEFE', '#FDEBD0', '#F5B041', '#E74C3C', '#7B241C'])
    im = ax.imshow(matrix_log, cmap=cmap, aspect='auto', interpolation='nearest')

    ax.set_xticks(range(len(top_t)))
    ax.set_yticks(range(len(top_s)))
    ax.set_xticklabels(top_t, rotation=50, ha='right', fontsize=6.5)
    ax.set_yticklabels(top_s, fontsize=6.5)
    ax.set_xlabel('Target odor', fontsize=7.5)
    ax.set_ylabel('Source odor', fontsize=7.5)
    ax.set_title('Transformation heatmap (log scale)', fontsize=9, fontweight='bold', pad=4)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('log(1 + freq)', fontsize=6.5, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=6)


def panel_D_top_ec_sequences(ax, data):
    """(D) Top EC sequences — horizontal bars color-coded by EC class."""
    N = 15
    top_ecs = data.get('top_ec_sequences', [])[:N]
    labels = [' \u2192 '.join(item.get('ec_sequence', [])) for item in top_ecs]
    freqs  = [float(item.get('weighted_frequency', 0)) for item in top_ecs]

    colors = []
    for item in top_ecs:
        seq = item.get('ec_sequence', [])
        cat = ec_category(seq[0]) if seq else 'Unknown'
        colors.append(EC_COLORS.get(cat, '#95A5A6'))

    ax.barh(range(len(labels)), freqs, color=colors, edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel('Weighted freq.', fontsize=7.5)
    ax.set_title('Top EC sequences', fontsize=9, fontweight='bold', pad=4)
    ax.grid(axis='x', alpha=0.2, linewidth=0.5)

    # Legend
    used = []
    for item in top_ecs:
        seq = item.get('ec_sequence', [])
        cat = ec_category(seq[0]) if seq else 'Unknown'
        if cat not in used:
            used.append(cat)
    patches = [mpatches.Patch(color=EC_COLORS.get(c, '#95A5A6'), label=c) for c in used]
    ax.legend(handles=patches, loc='lower right', fontsize=5.5, framealpha=0.8)


def panel_E_ec_pie(ax, data):
    """(E) EC functional class distribution (normalized by pathway length)."""
    top_ecs = data.get('top_ec_sequences', [])[:100]
    cat_weights = defaultdict(float)

    for ec_item in top_ecs:
        seq = ec_item.get('ec_sequence', [])
        freq = float(ec_item.get('weighted_frequency', 0))
        if not seq:
            continue
        w = freq / len(seq)
        for ec in seq:
            cat_weights[ec_category(ec)] += w

    if not cat_weights:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    pairs = sorted(cat_weights.items(), key=lambda x: -x[1])
    cats = [k for k, _ in pairs]
    vals = [v for _, v in pairs]
    colors = [EC_COLORS.get(c, '#95A5A6') for c in cats]

    wedges, texts, autotexts = ax.pie(
        vals, labels=cats, autopct='%1.1f%%', colors=colors,
        startangle=90, pctdistance=0.75,
        textprops={'fontsize': 6.5}, wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})

    for at in autotexts:
        at.set_fontsize(6)
        at.set_fontweight('bold')

    ax.set_title('EC class distribution\n(pathway-normalized)', fontsize=9, fontweight='bold', pad=2)


def panel_F_transformation_flow(ax, data):
    """(F) Sankey-like transformation flow."""
    triplets_all = data.get('top_triplets', [])
    filtered = [t for t in triplets_all if float(t.get('weighted_frequency', 0)) > 10]
    top_triplets = (filtered if filtered else triplets_all)[:30]

    sources = list(dict.fromkeys(t.get('source_odor', '') for t in top_triplets))[:8]
    targets = list(dict.fromkeys(t.get('target_odor', '') for t in top_triplets))[:8]

    relevant = [t for t in top_triplets
                if t.get('source_odor', '') in sources and t.get('target_odor', '') in targets]

    if not sources or not targets:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    y_s = np.linspace(0.05, 0.95, len(sources))
    y_t = np.linspace(0.05, 0.95, len(targets))
    sp = {s: y for s, y in zip(sources, y_s)}
    tp = {t: y for t, y in zip(targets, y_t)}

    max_freq = max((float(t.get('weighted_frequency', 1)) for t in relevant), default=1)

    for tr in relevant:
        s = tr.get('source_odor', '')
        t = tr.get('target_odor', '')
        freq = float(tr.get('weighted_frequency', 0))
        if s in sp and t in tp:
            x = np.linspace(0.18, 0.82, 80)
            # Smooth S-curve
            t_norm = (x - 0.18) / (0.82 - 0.18)
            y = sp[s] + (tp[t] - sp[s]) * (3*t_norm**2 - 2*t_norm**3)

            lw = 0.8 + 2.5 * (freq / max_freq)
            alpha = max(0.15, min(0.75, freq / max_freq))
            ax.plot(x, y, color='#2C3E50', alpha=alpha, lw=lw, solid_capstyle='round')

    for s, y in sp.items():
        ax.text(0.14, y, s, ha='right', va='center', fontsize=6.5, fontweight='bold',
                color='#2471A3')
    for t, y in tp.items():
        ax.text(0.86, y, t, ha='left', va='center', fontsize=6.5, fontweight='bold',
                color='#C0392B')

    # Vertical lines
    ax.axvline(0.17, color='#2471A3', alpha=0.3, lw=0.5, ls='-')
    ax.axvline(0.83, color='#C0392B', alpha=0.3, lw=0.5, ls='-')
    ax.text(0.14, 1.0, 'Source', ha='right', va='bottom', fontsize=7,
            color='#2471A3', fontweight='bold', transform=ax.transAxes)
    ax.text(0.86, 1.0, 'Target', ha='left', va='bottom', fontsize=7,
            color='#C0392B', fontweight='bold', transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.axis('off')
    ax.set_title('Major transformation flows', fontsize=9, fontweight='bold', pad=4)


def panel_G_rule_statistics(ax, rules_data):
    """(G) Rule type counts — horizontal bar chart."""
    if rules_data is None:
        ax.text(0.5, 0.5, 'No rule data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return

    # Count rules by type
    type_counts = defaultdict(int)
    for r in rules_data:
        rtype = r.get('rule_type', 'unknown')
        type_counts[rtype] += 1

    # Order by count descending
    sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
    names = [t for t, _ in sorted_types]
    counts = [c for _, c in sorted_types]

    # Nicer display names
    display_names = {
        'exclusion': 'Exclusion',
        'disjunctive_source': 'Disjunctive src',
        'conditional_necessary': 'Cond. necessary',
        'mutual_exclusion': 'Mutual exclusion',
        'necessary': 'Necessary',
        'sufficient': 'Sufficient',
    }
    labels = [display_names.get(n, n) for n in names]

    rule_colors = ['#E74C3C', '#3498DB', '#F39C12', '#9B59B6', '#27AE60', '#1ABC9C']
    bar_colors = [rule_colors[i % len(rule_colors)] for i in range(len(labels))]

    bars = ax.barh(range(len(labels)), counts, color=bar_colors,
                   edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()

    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + max(counts)*0.02, bar.get_y() + bar.get_height()/2,
                f'{c:,}', ha='left', va='center', fontsize=6.5, fontweight='bold')

    ax.set_xlabel('Count', fontsize=7.5)
    ax.set_title('Rule type distribution', fontsize=9, fontweight='bold', pad=4)
    ax.set_xlim(0, max(counts) * 1.2 if counts else 1)
    ax.grid(axis='x', alpha=0.2, linewidth=0.5)


def panel_H_data_scale(ax, data):
    """(H) Data scale summary — compact stats panel."""
    summary = data.get('summary', {})

    metrics = [
        ('Odorous compounds', 542),  # from paper
        ('Odor attributes', 138),
        ('Pathways', summary.get('pathways', 0)),
        ('Odor events', summary.get('odor_events', 0)),
        ('Unique triplets', summary.get('unique_triplets', 0)),
        ('EC sequences', summary.get('unique_ec_sequences', 0)),
    ]

    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]

    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(values)))
    bars = ax.barh(range(len(labels)), values, color=colors,
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values)*0.02, bar.get_y() + bar.get_height()/2,
                f'{v:,}', ha='left', va='center', fontsize=6.5, fontweight='bold')

    expansion = summary.get('expansion_ratio', 0)
    if expansion:
        ax.text(0.97, 0.05, f'Expansion: {expansion:.1f}\u00d7',
                transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF9E7', alpha=0.8))

    ax.set_xlabel('Count', fontsize=7.5)
    ax.set_title('Dataset scale', fontsize=9, fontweight='bold', pad=4)
    ax.set_xlim(0, max(values) * 1.18 if max(values) > 0 else 1)
    ax.grid(axis='x', alpha=0.2, linewidth=0.5)


# ═══════════════════════════════════════════════════════════════════
#  MAIN: Assemble the composite figure
# ═══════════════════════════════════════════════════════════════════

def build_composite_figure(data, rules_data, output_dir):
    """
    Build an 8-panel composite figure with intelligent layout:

    ┌──────────────────┬───────────────────┬────────────────┐
    │ (A) Odor distrib │ (B) Path length   │ (C) Heatmap    │
    │  (src | tgt)     │                   │                │
    ├──────────────────┼───────────────────┼────────────────┤
    │ (D) Top EC seq   │ (E) EC class pie  │ (F) Flow       │
    │                  │                   │                │
    ├──────────────────┴───────────────────┴────────────────┤
    │ (G) Rule statistics        │ (H) Data scale           │
    └────────────────────────────┴──────────────────────────┘
    """
    fig = plt.figure(figsize=(17, 16.5))

    # Outer grid: 3 rows, heights ratio 5:5:3
    outer = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[5, 5, 3.2],
                              hspace=0.32)

    # ── Row 1: A (split), B, C ──────────────────────────────────
    row1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0],
                                            wspace=0.38, width_ratios=[1.1, 0.9, 1.0])

    # A: Source + Target side by side
    inner_A = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=row1[0], wspace=0.45)
    ax_A1 = fig.add_subplot(inner_A[0])
    ax_A2 = fig.add_subplot(inner_A[1])
    panel_A_odor_distribution(ax_A1, ax_A2, data)

    ax_B = fig.add_subplot(row1[1])
    panel_B_pathway_length(ax_B, data)

    ax_C = fig.add_subplot(row1[2])
    panel_C_heatmap(ax_C, data)

    # ── Row 2: D, E, F ─────────────────────────────────────────
    row2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1],
                                            wspace=0.35, width_ratios=[1.1, 0.8, 1.1])
    ax_D = fig.add_subplot(row2[0])
    panel_D_top_ec_sequences(ax_D, data)

    ax_E = fig.add_subplot(row2[1])
    panel_E_ec_pie(ax_E, data)

    ax_F = fig.add_subplot(row2[2])
    panel_F_transformation_flow(ax_F, data)

    # ── Row 3: G, H ────────────────────────────────────────────
    row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2],
                                            wspace=0.35, width_ratios=[1, 1])
    ax_G = fig.add_subplot(row3[0])
    panel_G_rule_statistics(ax_G, rules_data)

    ax_H = fig.add_subplot(row3[1])
    panel_H_data_scale(ax_H, data)

    # ── Panel labels (A)–(H) ───────────────────────────────────
    for ax, label in [(ax_A1, 'A'), (ax_B, 'B'), (ax_C, 'C'),
                      (ax_D, 'D'), (ax_E, 'E'), (ax_F, 'F'),
                      (ax_G, 'G'), (ax_H, 'H')]:
        ax.text(-0.08, 1.08, f'({label})', transform=ax.transAxes, **PANEL_LABEL_KW)

    fig.suptitle('Empirical Overview of Odor Transformation Rules',
                 fontsize=16, fontweight='bold', y=0.995)

    # ── Save ────────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    png_path = out / 'fig3_empirical_dashboard.png'
    pdf_path = out / 'fig3_empirical_dashboard.pdf'

    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'\u2713 Saved: {png_path}')
    print(f'\u2713 Saved: {pdf_path}')
    return png_path, pdf_path


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Generate composite Fig.3')
    parser.add_argument('--json', default='../03_rule_extraction/odor_level_patterns_weighted.json',
                        help='Path to viz data JSON')
    parser.add_argument('--rules', default='../03_rule_extraction/complex_fol_rules.json',
                        help='Path to rules JSON')
    parser.add_argument('--outdir', default='figures', help='Output directory')
    args = parser.parse_args()

    # Load viz data
    json_path = Path(args.json)
    if not json_path.exists():
        print(f'ERROR: {json_path} not found.')
        print('Run the v5 pipeline first:')
        print('  cd 03_rule_extraction && python v5_all_with_vis.py')
        return

    with open(json_path) as f:
        data = json.load(f)
    print(f'Loaded viz data: {len(data.get("top_triplets", []))} triplets, '
          f'{len(data.get("top_ec_sequences", []))} EC sequences')

    # Load rules (optional)
    rules_path = Path(args.rules)
    rules_data = None
    if rules_path.exists():
        with open(rules_path) as f:
            rules_data = json.load(f)
        print(f'Loaded rules: {len(rules_data)} rules')
    else:
        print(f'Warning: {rules_path} not found, skipping rule panel')

    random.seed(42)
    build_composite_figure(data, rules_data, args.outdir)


if __name__ == '__main__':
    main()
