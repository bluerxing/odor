"""
v5_plots.py — All visualization functions (Figures 1–11)

Optimizations vs. original:
  - matplotlib Agg backend (via v5_config import)
  - Default DPI 150 for fast drafts; pass dpi=300 for publication
  - plt.close() on every figure to free memory
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter, defaultdict

# Import config to ensure Agg backend + rcParams are set
import v5_config  # noqa: F401

# Default DPI — override with `set_dpi(300)` for publication quality
_DPI = 150


def set_dpi(dpi: int):
    """Set default DPI for all subsequent plots."""
    global _DPI
    _DPI = dpi


def _save(fig, path, **kwargs):
    """Save and close a figure."""
    fig.savefig(path, dpi=_DPI, bbox_inches='tight', **kwargs)
    plt.close(fig)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def get_ec_category(ec_number):
    categories = {
        '1': 'Oxidoreductase', '2': 'Transferase', '3': 'Hydrolase',
        '4': 'Lyase', '5': 'Isomerase', '6': 'Ligase'
    }
    if ec_number and '.' in ec_number:
        return categories.get(ec_number.split('.')[0], 'Unknown')
    return 'Unknown'


EC_COLOR_MAP = {
    'Oxidoreductase': '#FF6B6B', 'Transferase': '#4ECDC4',
    'Hydrolase': '#45B7D1', 'Lyase': '#FFA07A',
    'Isomerase': '#98D8C8', 'Ligase': '#F7DC6F',
    'Unknown': '#B0B0B0', 'No Data': '#CCCCCC'
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ===============================================================
# Individual figures
# ===============================================================

def plot_1_data_overview(data, output_dir):
    """Figure 1: Data Scale Pyramid"""
    fig, ax = plt.subplots(figsize=(10, 6))
    summary = data.get('summary', {})
    stages = ['Pathways', 'Odor Events\n(Cartesian)', 'Unique Triplets', 'Unique EC\nSequences']
    values = [summary.get('pathways', 0), summary.get('odor_events', 0),
              summary.get('unique_triplets', 0), summary.get('unique_ec_sequences', 0)]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(values)))
    bars = ax.barh(stages, values, color=colors, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', ha='left', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Data Scale: From Pathways to Patterns', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, max(values) * 1.15 if max(values) > 0 else 1)
    er = summary.get('expansion_ratio', 0.0)
    ax.text(0.98, 0.05, f"Expansion Ratio: {er:.2f}x", transform=ax.transAxes,
            fontsize=10, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.tight_layout()
    _save(fig, Path(output_dir) / '01_data_overview.png')
    print("✓ Figure 1: Data Scale Pyramid")


def plot_2_odor_distribution(data, output_dir):
    """Figure 2: Odor Category Distribution (Source vs Target)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, key, color, title in [
        (ax1, 'top_source_odors', 'steelblue', 'Top 15 Source Odors'),
        (ax2, 'top_target_odors', 'coral', 'Top 15 Target Odors'),
    ]:
        items = data.get(key, [])[:15]
        names = [d.get('odor', '') for d in items]
        weights = [d.get('total_weight', 0.0) for d in items]
        ax.barh(names, weights, color=color, edgecolor='black')
        ax.set_xlabel('Weighted Frequency', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.invert_yaxis()

    fig.suptitle('Odor Category Distribution', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, Path(output_dir) / '02_odor_distribution.png')
    print("✓ Figure 2: Odor Category Distribution")


def plot_3_pathway_length_distribution(data, output_dir):
    """Figure 3: EC Sequence Length Distribution"""
    lengths = []
    top_triplets = data.get('top_triplets', [])
    top_ec_sequences = data.get('top_ec_sequences', [])

    if top_triplets:
        sample_size = min(10000, len(top_triplets))
        sampled = random.sample(top_triplets, sample_size) if len(top_triplets) > sample_size else top_triplets
        lengths = [len(t.get('ec_sequence', [])) for t in sampled]
        source_tag = f"Sampled {len(sampled):,} triplets"
    elif top_ec_sequences:
        lengths = [len(item.get('ec_sequence', [])) for item in top_ec_sequences]
        source_tag = f"From top_ec_sequences ({len(top_ec_sequences)})"
    else:
        source_tag = "No sequence data available"

    fig, ax = plt.subplots(figsize=(10, 6))
    if lengths:
        counts = Counter(lengths)
        sorted_lengths = sorted(counts.keys())
        freqs = [counts[l] for l in sorted_lengths]
        bars = ax.bar(sorted_lengths, freqs, color='mediumseagreen', edgecolor='black', alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h, f'{int(h)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(sorted_lengths)
        textstr = f'{source_tag}\nMean: {np.mean(lengths):.1f}\nMedian: {np.median(lengths):.1f}'
    else:
        ax.bar([0], [0], color='mediumseagreen', edgecolor='black', alpha=0.85)
        ax.set_xticks([])
        textstr = source_tag

    ax.set_xlabel('Pathway Length (# of EC steps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('EC Sequence Length Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    fig.tight_layout()
    _save(fig, Path(output_dir) / '03_pathway_length.png')
    print("✓ Figure 3: EC Sequence Length Distribution")


def plot_4_odor_transformation_heatmap(data, output_dir):
    """Figure 4: Odor Transformation Heatmap"""
    top_sources = [d.get('odor', '') for d in data.get('top_source_odors', [])[:15]]
    top_targets = [d.get('odor', '') for d in data.get('top_target_odors', [])[:15]]
    matrix = np.zeros((len(top_sources), len(top_targets)))
    top_triplets = data.get('top_triplets', [])
    use_n = min(5000, len(top_triplets))

    # Build lookup for faster matrix filling
    src_idx = {s: i for i, s in enumerate(top_sources)}
    tgt_idx = {t: j for j, t in enumerate(top_targets)}
    for triplet in top_triplets[:use_n]:
        s, t = triplet.get('source_odor', ''), triplet.get('target_odor', '')
        if s in src_idx and t in tgt_idx:
            matrix[src_idx[s], tgt_idx[t]] += float(triplet.get('weighted_frequency', 0.0))

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(np.arange(len(top_targets)))
    ax.set_yticks(np.arange(len(top_sources)))
    ax.set_xticklabels(top_targets, rotation=45, ha='right')
    ax.set_yticklabels(top_sources)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weighted Frequency', rotation=270, labelpad=20, fontweight='bold')
    ax.set_title(f'Odor Transformation Heatmap (Top {use_n} Triplets)\n(Source → Target)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Target Odor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Source Odor', fontsize=12, fontweight='bold')
    fig.tight_layout()
    _save(fig, Path(output_dir) / '04_transformation_heatmap.png')
    print("✓ Figure 4: Odor Transformation Heatmap")


def plot_5_top_ec_sequences(data, output_dir):
    """Figure 5: Top EC Sequences Bar Chart"""
    fig, ax = plt.subplots(figsize=(12, 10))
    top_ecs = data.get('top_ec_sequences', [])[:20]
    ec_labels = [' → '.join(item.get('ec_sequence', [])) for item in top_ecs]
    freqs = [float(item.get('weighted_frequency', 0.0)) for item in top_ecs]

    used_cats = []
    colors = []
    for item in top_ecs:
        seq = item.get('ec_sequence', [])
        cat = get_ec_category(seq[0] if seq else '')
        colors.append(EC_COLOR_MAP.get(cat, '#B0B0B0'))
        used_cats.append(cat)

    ax.barh(range(len(ec_labels)), freqs, color=colors, edgecolor='black', linewidth=1)
    ax.set_yticks(range(len(ec_labels)))
    ax.set_yticklabels(ec_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Weighted Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 EC Sequences (Odor Space)', fontsize=14, fontweight='bold', pad=20)

    unique_used = list(dict.fromkeys(used_cats))
    legend_patches = [mpatches.Patch(color=EC_COLOR_MAP.get(c, '#B0B0B0'), label=c) for c in unique_used]
    if legend_patches:
        ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
    fig.tight_layout()
    _save(fig, Path(output_dir) / '05_top_ec_sequences.png')
    print("✓ Figure 5: Top EC Sequences")


def plot_6_sankey_preview(data, output_dir):
    """Figure 6: Odor Transformation Flow (Sankey-like)"""
    fig, ax = plt.subplots(figsize=(14, 8))
    top_triplets_all = data.get('top_triplets', [])
    filtered = [t for t in top_triplets_all if float(t.get('weighted_frequency', 0.0)) > 10]
    top_triplets = (filtered if filtered else top_triplets_all)[:30]
    sources = list({t.get('source_odor', '') for t in top_triplets})[:8]
    targets = list({t.get('target_odor', '') for t in top_triplets})[:8]
    filtered = [t for t in top_triplets if t.get('source_odor', '') in sources and t.get('target_odor', '') in targets]

    y_source = np.linspace(0, 1, len(sources)) if sources else []
    y_target = np.linspace(0, 1, len(targets)) if targets else []
    source_pos = dict(zip(sources, y_source))
    target_pos = dict(zip(targets, y_target))

    for triplet in filtered:
        s, t = triplet.get('source_odor', ''), triplet.get('target_odor', '')
        freq = float(triplet.get('weighted_frequency', 0.0))
        if s in source_pos and t in target_pos:
            y1, y2 = source_pos[s], target_pos[t]
            x = np.linspace(0.2, 0.8, 100)
            y = y1 + (y2 - y1) * (3 * x**2 - 2 * x**3)
            width = 1.0 + 0.3 * np.sqrt(max(freq, 0.0))
            alpha = _clamp(freq / 50.0, 0.1, 0.8)
            ax.plot(x, y, alpha=alpha, linewidth=width, color='steelblue')

    for s, y in source_pos.items():
        ax.text(0.15, y, s, ha='right', va='center', fontsize=10, fontweight='bold')
    for t, y in target_pos.items():
        ax.text(0.85, y, t, ha='left', va='center', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(-0.1, 1.1); ax.axis('off')
    ax.set_title('Major Odor Transformation Flows\n(Line width ∝ sqrt(frequency))',
                 fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    _save(fig, Path(output_dir) / '06_transformation_flow.png')
    print("✓ Figure 6: Transformation Flow")


def _plot_ec_by_odor(data, output_dir, odor_key, cmap, filename, title_suffix):
    """Shared logic for Figures 7a/7b."""
    top_ecs = data.get('top_ec_sequences', [])[:10]
    top_odors = [d.get('odor', '') for d in data.get(odor_key, [])[:15]]
    matrix = np.zeros((len(top_ecs), len(top_odors)))
    top_triplets = data.get('top_triplets', [])[:1000]

    # Build ec lookup for speed
    ec_tuples = [tuple(item.get('ec_sequence', [])) for item in top_ecs]
    odor_idx = {o: j for j, o in enumerate(top_odors)}
    odor_field = 'source_odor' if 'source' in odor_key else 'target_odor'

    for i, ec_seq in enumerate(ec_tuples):
        for triplet in top_triplets:
            if tuple(triplet.get('ec_sequence', [])) == ec_seq:
                o = triplet.get(odor_field, '')
                if o in odor_idx:
                    matrix[i, odor_idx[o]] += float(triplet.get('weighted_frequency', 0.0))

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    ec_labels = [(' → '.join(item.get('ec_sequence', [])[:3]) + '...'
                  if len(item.get('ec_sequence', [])) > 3
                  else ' → '.join(item.get('ec_sequence', []))) for item in top_ecs]
    ax.set_xticks(np.arange(len(top_odors)))
    ax.set_yticks(np.arange(len(top_ecs)))
    ax.set_xticklabels(top_odors, rotation=45, ha='right')
    ax.set_yticklabels(ec_labels, fontsize=9)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weighted Frequency', rotation=270, labelpad=20, fontweight='bold')
    ax.set_title(f'EC Sequence Usage by {title_suffix}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(title_suffix, fontsize=12, fontweight='bold')
    ax.set_ylabel('EC Sequence', fontsize=12, fontweight='bold')
    fig.tight_layout()
    _save(fig, Path(output_dir) / filename)


def plot_7_ec_by_odor_category_source(data, output_dir):
    _plot_ec_by_odor(data, output_dir, 'top_source_odors', 'Greens',
                     '07a_ec_by_source.png', 'Source Odor Category')
    print("✓ Figure 7a: EC × Source Odor")


def plot_7_ec_by_odor_category_target(data, output_dir):
    _plot_ec_by_odor(data, output_dir, 'top_target_odors', 'Purples',
                     '07b_ec_by_target.png', 'Target Odor Category')
    print("✓ Figure 7b: EC × Target Odor")


def _aggregate_ec_categories(top_ec_sequences, normalized=True):
    ec_categories = defaultdict(float)
    for item in top_ec_sequences:
        seq = item.get('ec_sequence', [])
        freq = float(item.get('weighted_frequency', 0.0))
        if not seq:
            continue
        if normalized:
            w = freq / len(seq)
            for ec in seq:
                ec_categories[get_ec_category(ec)] += w
        else:
            for ec in seq:
                ec_categories[get_ec_category(ec)] += freq
    return ec_categories


def _plot_pie_from_categories(ec_categories, title, outpath):
    fig, ax = plt.subplots(figsize=(10, 8))
    pairs = sorted([(k, v) for k, v in ec_categories.items() if v > 0], key=lambda x: -x[1])
    if not pairs:
        pairs = [('No Data', 1.0)]
    cats = [k for k, _ in pairs]
    vals = [v for _, v in pairs]
    colors = [EC_COLOR_MAP.get(c, '#B0B0B0') for c in cats]
    wedges, texts, autotexts = ax.pie(
        vals, labels=cats, autopct='%1.1f%%', colors=colors, startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'})
    for at in autotexts:
        at.set_color('white'); at.set_fontsize(12); at.set_fontweight('bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    _save(fig, outpath)


def plot_8_ec_function_pies(data, output_dir):
    top_ecs = data.get('top_ec_sequences', [])[:100]
    _plot_pie_from_categories(
        _aggregate_ec_categories(top_ecs, normalized=True),
        'EC Function Distribution (Normalized by Pathway Length)',
        Path(output_dir) / '08a_ec_function_pie_normalized.png')
    _plot_pie_from_categories(
        _aggregate_ec_categories(top_ecs, normalized=False),
        'EC Function Distribution (By Step / Raw Accumulation)',
        Path(output_dir) / '08b_ec_function_pie_by_step.png')
    print("✓ Figure 8a/8b: EC Function Pies")


def plot_9_case_studies(data, output_dir):
    """Figure 9: Representative Case Studies"""
    triplets = data.get('top_triplets', [])
    if not triplets:
        print("! Figure 9 skipped (no triplets)")
        return

    case_indices = [i for i in [0, 5, 15] if i < len(triplets)]
    if not case_indices:
        case_indices = list(range(min(3, len(triplets))))

    fig, axes = plt.subplots(len(case_indices), 1, figsize=(12, 10))
    if len(case_indices) == 1:
        axes = [axes]

    for ax_idx, ax in enumerate(axes):
        triplet = triplets[case_indices[ax_idx]]
        s_odor, t_odor = triplet.get('source_odor', ''), triplet.get('target_odor', '')
        ec_seq = triplet.get('ec_sequence', [])
        freq = float(triplet.get('weighted_frequency', 0.0))
        examples = triplet.get('examples', [])
        n_steps = len(ec_seq)
        x_pos = np.linspace(0, 1, n_steps + 2)

        for i in range(n_steps):
            ax.annotate('', xy=(x_pos[i + 1], 0.5), xytext=(x_pos[i], 0.5),
                        arrowprops=dict(arrowstyle='->', lw=3))
            ax.text((x_pos[i] + x_pos[i + 1]) / 2, 0.7, ec_seq[i],
                    ha='center', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        ax.text(x_pos[0], 0.5, s_odor, ha='right', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.text(x_pos[-1], 0.5, t_odor, ha='left', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax.set_title(f"Case {ax_idx + 1}: {s_odor} → {t_odor} (freq: {freq:.1f})",
                     fontsize=11, fontweight='bold', loc='left')
        if examples:
            ex_strs = [f"{e.get('src_compound', '')}→{e.get('tgt_compound', '')}" for e in examples[:2]]
            ax.text(0.5, 0.1, f"Examples: {', '.join(ex_strs)}", ha='center', va='top',
                    fontsize=9, style='italic', transform=ax.transAxes)
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(0, 1); ax.axis('off')

    fig.suptitle('Representative Odor Transformation Pathways', fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    _save(fig, Path(output_dir) / '09_case_studies.png')
    print("✓ Figure 9: Case Studies")


def plot_10_key_findings(data, output_dir):
    """Figure 10: Key Findings Summary"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # Top-3 transformations
    ax1 = fig.add_subplot(gs[0, 0])
    top3 = data.get('top_triplets', [])[:3]
    labels = [f"{t.get('source_odor', '')}→{t.get('target_odor', '')}" for t in top3]
    values = [float(t.get('weighted_frequency', 0.0)) for t in top3]
    ax1.barh(labels, values, color=['gold', 'silver', '#CD7F32'], edgecolor='black')
    ax1.set_xlabel('Frequency', fontweight='bold')
    ax1.set_title('[Top 3] Transformations', fontweight='bold')
    ax1.invert_yaxis()

    # Most common EC
    ax2 = fig.add_subplot(gs[0, 1])
    top_ec = data.get('top_ec_sequences', [{}])[0]
    ax2.text(0.5, 0.6, ' → '.join(top_ec.get('ec_sequence', [])) or 'N/A',
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax2.text(0.5, 0.3, f"Frequency: {float(top_ec.get('weighted_frequency', 0.0)):.0f}",
             ha='center', va='center', fontsize=10)
    ax2.set_title('[Most Common] EC Sequence', fontweight='bold')
    ax2.axis('off')

    # Coverage stats
    ax3 = fig.add_subplot(gs[1, 0])
    s = data.get('summary', {})
    ax3.text(0.1, 0.5, (f"  Pathways: {s.get('pathways', 0):,}\n"
                         f"  Unique Odor Pairs: {s.get('unique_triplets', 0):,}\n"
                         f"  Source Odors: {s.get('unique_source_odors', 0)}\n"
                         f"  Target Odors: {s.get('unique_target_odors', 0)}\n"
                         f"  EC Sequences: {s.get('unique_ec_sequences', 0):,}"),
             ha='left', va='center', fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax3.set_title('[Statistics] Coverage', fontweight='bold')
    ax3.axis('off')

    # Bio insights
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.text(0.1, 0.5,
             "  Key Insights:\n  • Oxidoreductases (EC 1.x) dominate\n"
             "  • 2-3 step pathways most common\n  • Phenolic→Woody appears prominently\n"
             "  • Mint/Floral act as hubs",
             ha='left', va='center', fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax4.set_title('[Biology] Key Insights', fontweight='bold')
    ax4.axis('off')

    # Notable lower-freq
    ax5 = fig.add_subplot(gs[2, :])
    triplets = data.get('top_triplets', [])
    rare = [t for t in triplets[20:50] if float(t.get('weighted_frequency', 0.0)) < 15][:6]
    if rare:
        labels = [f"{t.get('source_odor', '')}→{t.get('target_odor', '')}\n"
                  f"({' → '.join(t.get('ec_sequence', [])[:2])}...)" for t in rare]
        vals = [float(t.get('weighted_frequency', 0.0)) for t in rare]
        ax5.bar(range(len(labels)), vals, color='mediumpurple', edgecolor='black', alpha=0.7)
        ax5.set_xticks(range(len(labels)))
        ax5.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('[Notable] Specific Transformations (Lower frequency)', fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No notable low-frequency cases found.', ha='center', va='center')
        ax5.axis('off')

    fig.suptitle('Key Findings Summary', fontsize=16, fontweight='bold', y=0.98)
    _save(fig, Path(output_dir) / '10_key_findings.png')
    print("✓ Figure 10: Key Findings")


def plot_11_empirical_dashboard(output_dir):
    """Figure 11: Paper-ready composite dashboard (2x3 panels)."""
    output_dir = Path(output_dir)
    panel_specs = [
        ("A", "Odor Distribution", "02_odor_distribution.png"),
        ("B", "Pathway Length", "03_pathway_length.png"),
        ("C", "Transformation Heatmap", "04_transformation_heatmap.png"),
        ("D", "Top EC Sequences", "05_top_ec_sequences.png"),
        ("E", "EC by Source Odor", "07a_ec_by_source.png"),
        ("F", "EC by Target Odor", "07b_ec_by_target.png"),
    ]
    missing = [name for _, _, name in panel_specs if not (output_dir / name).exists()]
    if missing:
        print(f"! Figure 11 skipped, missing panels: {', '.join(missing)}")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    for ax, (tag, title, filename) in zip(axes, panel_specs):
        img = plt.imread(output_dir / filename)
        ax.imshow(img); ax.set_axis_off()
        ax.set_title(f"({tag}) {title}", fontsize=12, fontweight='bold', pad=8)
    fig.suptitle("Empirical Overview of Odor Transformation Rules",
                 fontsize=18, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    _save(fig, output_dir / '11_empirical_dashboard.png')
    fig2, axes2 = plt.subplots(3, 2, figsize=(16, 18))
    axes2 = axes2.flatten()
    for ax, (tag, title, filename) in zip(axes2, panel_specs):
        img = plt.imread(output_dir / filename)
        ax.imshow(img); ax.set_axis_off()
        ax.set_title(f"({tag}) {title}", fontsize=12, fontweight='bold', pad=8)
    fig2.suptitle("Empirical Overview of Odor Transformation Rules",
                  fontsize=18, fontweight='bold', y=0.995)
    fig2.tight_layout(rect=[0, 0, 1, 0.985])
    fig2.savefig(output_dir / '11_empirical_dashboard.pdf', bbox_inches='tight')
    plt.close(fig2)
    print("✓ Figure 11: Empirical Dashboard (PNG + PDF)")


# ===============================================================
# Convenience: run all plots
# ===============================================================

def run_all_plots(viz_data, output_dir):
    """Generate all figures 1-11 from viz_data dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    plot_1_data_overview(viz_data, output_dir)
    plot_2_odor_distribution(viz_data, output_dir)
    plot_3_pathway_length_distribution(viz_data, output_dir)
    plot_4_odor_transformation_heatmap(viz_data, output_dir)
    plot_5_top_ec_sequences(viz_data, output_dir)
    plot_6_sankey_preview(viz_data, output_dir)
    plot_7_ec_by_odor_category_source(viz_data, output_dir)
    plot_7_ec_by_odor_category_target(viz_data, output_dir)
    plot_8_ec_function_pies(viz_data, output_dir)
    plot_9_case_studies(viz_data, output_dir)
    plot_10_key_findings(viz_data, output_dir)
    plot_11_empirical_dashboard(output_dir)

    print(f"\n✅ 所有图表已保存到: {output_dir}/")
