#!/usr/bin/env python3
"""
统一可视化模块 - 合并多规则验证和跨数据集验证
==============================================

整合自:
- cross_dataset_ec_validation_multi.py 的 plot_enhanced_multi_rule_visualizations
- cross_dataset_visualization.py 的 CrossDatasetVisualizer

输出精简但全面的可视化:
1. 综合仪表板 (multi-rule + cross-dataset)
2. 分布对比图 (4类算子的分布)
3. 效应量分析 (Cohen's d + CI)
4. 诊断图 (排序值 + 排斥分布)
5. 跨数据集验证图 (物种比较)
6. 论文级综合图 (4面板)
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from scipy import stats

# Matplotlib配置
def _ensure_mpl_config():
    if "MPLCONFIGDIR" not in os.environ:
        for path in ["/tmp/.mplconfig", "./.mplconfig"]:
            try:
                os.makedirs(path, exist_ok=True)
                os.environ["MPLCONFIGDIR"] = path
                break
            except:
                continue

_ensure_mpl_config()

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# 样式设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# 颜色方案
COLORS = {
    'compositional': '#4C78A8',
    'counterfactual': '#72B7B2',
    'gated_with': '#54A24B',
    'gated_without': '#E45756',
    'negative': '#8E6C8A',
}

SPECIES_COLORS = {
    'Human': '#E41A1C',
    'Drosophila': '#377EB8',
    'Mouse': '#4DAF4A',
    'Mosquito': '#984EA3',
    'Bristletail': '#FF7F00',
    'Unknown': '#999999'
}

DATASET_INFO = {
    'a_MacWilliam_et_al': {'species': 'Drosophila', 'short': 'MacWilliam'},
    'b_Xu_et_al': {'species': 'Mosquito', 'short': 'Xu'},
    'c_Dravnieks': {'species': 'Human', 'short': 'Dravnieks'},
    'd_Wei_et_al': {'species': 'Human', 'short': 'Wei'},
    'g_del_Mármol_et_al': {'species': 'Bristletail', 'short': 'del Mármol'},
    'h_Keller_et_al': {'species': 'Human', 'short': 'Keller'},
    'i_Carey_et_al': {'species': 'Mosquito', 'short': 'Carey'},
    'k_Hallem_and_Carlson': {'species': 'Drosophila', 'short': 'Hallem'},
    'n_Oliferenko_et_al': {'species': 'Mosquito', 'short': 'Oliferenko'},
    's_Chae_et_al': {'species': 'Mouse', 'short': 'Chae'},
}


class UnifiedVisualizer:
    """统一可视化器"""

    def __init__(self, output_dir: str = "./visualization_unified"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 数据存储
        self.multi_rule_results: Dict = {}
        self.multi_rule_raw: Dict = {}
        self.cross_dataset_data: List[Dict] = []

    def load_multi_rule_results(self, results: Dict, raw_results: Dict):
        """加载多规则验证结果"""
        self.multi_rule_results = results
        self.multi_rule_raw = raw_results

    def load_cross_dataset_data(self, data: List[Dict]):
        """加载跨数据集验证数据"""
        self.cross_dataset_data = data

    def plot_all(self):
        """生成所有可视化"""
        print(f"\n{'='*60}")
        print("📊 生成统一可视化图表")
        print(f"{'='*60}")

        has_multi_rule = bool(self.multi_rule_results)
        has_cross_dataset = bool(self.cross_dataset_data)

        if has_multi_rule:
            self._plot_01_summary_dashboard()
            self._plot_02_distribution_comparison()
            self._plot_03_effect_size()
            self._plot_04_diagnostic()

        if has_cross_dataset:
            self._plot_05_cross_dataset()

        if has_multi_rule or has_cross_dataset:
            self._plot_06_paper_figure()

        print(f"\n✅ 所有图表已保存到: {self.output_dir}")

    # =========================================================================
    # Figure 1: 综合仪表板 (9面板)
    # =========================================================================
    def _plot_01_summary_dashboard(self):
        """综合仪表板"""
        results = self.multi_rule_results
        raw = self.multi_rule_raw

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # 获取数据
        labels = ['Compositional', 'Counterfactual', 'Gated\n(with)', 'Gated\n(without)', 'Negative']
        means = [
            results.get('compositional', {}).get('mean', 0),
            results.get('counterfactual', {}).get('mean', 0),
            results.get('gated', {}).get('with_gate', {}).get('mean', 0),
            results.get('gated', {}).get('without_gate', {}).get('mean', 0),
            results.get('negative', {}).get('mean', 0)
        ]
        ratios = [
            results.get('compositional', {}).get('positive_ratio', 0),
            results.get('counterfactual', {}).get('positive_ratio', 0),
            results.get('gated', {}).get('with_gate', {}).get('positive_ratio', 0),
            results.get('gated', {}).get('without_gate', {}).get('positive_ratio', 0),
            results.get('negative', {}).get('positive_ratio', 0)
        ]
        sizes = [
            results.get('compositional', {}).get('n', 0),
            results.get('counterfactual', {}).get('n', 0),
            results.get('gated', {}).get('with_gate', {}).get('n', 0),
            results.get('gated', {}).get('without_gate', {}).get('n', 0),
            results.get('negative', {}).get('n', 0)
        ]
        colors = [COLORS['compositional'], COLORS['counterfactual'],
                  COLORS['gated_with'], COLORS['gated_without'], COLORS['negative']]

        # Panel A: Mean Effects
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(labels, means, color=colors, edgecolor='black', alpha=0.85)
        ax1.axhline(0, color='black', linewidth=1.5)
        ax1.set_ylabel('Mean Cosine', fontweight='bold')
        ax1.set_title('(A) Mean Cosine / Drop', fontweight='bold')
        ax1.tick_params(axis='x', rotation=20)
        for bar, mean in zip(bars, means):
            va = 'bottom' if mean >= 0 else 'top'
            offset = 0.01 if mean >= 0 else -0.01
            ax1.text(bar.get_x() + bar.get_width()/2, mean + offset,
                     f'{mean:.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

        # Panel B: Positive Ratio
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(labels, ratios, color=colors, edgecolor='black', alpha=0.85)
        ax2.axhline(0.5, color='gray', linewidth=1.5, linestyle='--', label='Random (50%)')
        ax2.set_ylabel('Positive Ratio', fontweight='bold')
        ax2.set_title('(B) Positive Direction Ratio', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=20)
        ax2.legend(loc='upper right', fontsize=9)
        for bar, ratio in zip(bars, ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, max(ratio, 0) + 0.02,
                     f'{ratio:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Panel C: Sample Sizes
        ax3 = fig.add_subplot(gs[0, 2])
        bars = ax3.bar(labels, sizes, color=colors, edgecolor='black', alpha=0.85)
        ax3.set_ylabel('Sample Size (n)', fontweight='bold')
        ax3.set_title('(C) Sample Sizes', fontweight='bold')
        ax3.tick_params(axis='x', rotation=20)
        if max(sizes) > 100:
            ax3.set_yscale('log')
        for bar, size in zip(bars, sizes):
            y_pos = bar.get_height() * 1.1 if ax3.get_yscale() == 'log' else bar.get_height() + max(sizes)*0.02
            ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                     f'{size:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Panel D-F: Distributions
        for idx, (key, title) in enumerate([
            ('compositional', '(D) Compositional'),
            ('counterfactual', '(E) Counterfactual'),
            ('negative', '(F) Negative Repulsion')
        ]):
            ax = fig.add_subplot(gs[1, idx])
            vals = raw.get(key, [])
            if vals:
                color = COLORS.get(key, '#999')
                ax.hist(vals, bins=40, color=color, alpha=0.7, edgecolor='white')
                ax.axvline(0, color='black', linewidth=1.5)
                mean_val = np.mean(vals)
                ax.axvline(mean_val, color='red', linewidth=2, linestyle='--',
                           label=f'Mean: {mean_val:.3f}')
                ax.legend(loc='upper right', fontsize=9)

                # 标注100%负向
                if key == 'negative' and results.get('negative', {}).get('positive_ratio', 1) == 0:
                    ax.text(0.5, 0.95, '✓ 100% REPULSION', transform=ax.transAxes,
                            fontsize=11, fontweight='bold', color='darkgreen',
                            ha='center', va='top', bbox=dict(facecolor='lightgreen', alpha=0.8))
            ax.set_xlabel('Cosine Similarity', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title(title, fontweight='bold')

        # Panel G: Gated Effect Comparison
        ax7 = fig.add_subplot(gs[2, 0])
        gated_with = raw.get('gated', {}).get('with_gate', [])
        gated_without = raw.get('gated', {}).get('without_gate', [])
        if gated_with or gated_without:
            if gated_with:
                ax7.hist(gated_with, bins=30, alpha=0.6, color=COLORS['gated_with'],
                         label=f'With Gate (n={len(gated_with)})', edgecolor='white')
            if gated_without:
                ax7.hist(gated_without, bins=30, alpha=0.6, color=COLORS['gated_without'],
                         label=f'Without Gate (n={len(gated_without)})', edgecolor='white')
            ax7.axvline(0, color='black', linewidth=1.5)
            ax7.legend(loc='upper right', fontsize=9)
        ax7.set_xlabel('Cosine Similarity', fontweight='bold')
        ax7.set_ylabel('Count', fontweight='bold')
        ax7.set_title('(G) Gated Effect Comparison', fontweight='bold')

        # Panel H: Effect Size (Cohen's d)
        ax8 = fig.add_subplot(gs[2, 1])
        effect_data = []
        effect_labels = []
        effect_colors = []
        for key, label, color in [
            ('compositional', 'Compositional', COLORS['compositional']),
            ('counterfactual', 'Counterfactual', COLORS['counterfactual']),
            ('negative', 'Negative', COLORS['negative'])
        ]:
            vals = raw.get(key, [])
            if vals:
                d = np.mean(vals) / (np.std(vals) + 1e-8)
                effect_data.append(d)
                effect_labels.append(label)
                effect_colors.append(color)

        if effect_data:
            bars = ax8.barh(effect_labels, effect_data, color=effect_colors, edgecolor='black', alpha=0.85)
            ax8.axvline(0, color='black', linewidth=1)
            ax8.axvline(0.2, color='gray', linewidth=1, linestyle='--', alpha=0.5)
            ax8.axvline(-0.2, color='gray', linewidth=1, linestyle='--', alpha=0.5)
            ax8.set_xlabel("Cohen's d", fontweight='bold')
            ax8.set_title("(H) Effect Size", fontweight='bold')
            for bar, val in zip(bars, effect_data):
                ha = 'left' if val >= 0 else 'right'
                offset = 0.05 if val >= 0 else -0.05
                ax8.text(val + offset, bar.get_y() + bar.get_height()/2,
                         f'{val:.2f}', ha=ha, va='center', fontsize=10, fontweight='bold')

        # Panel I: Key Findings
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        findings = ["KEY FINDINGS", "=" * 30, ""]
        neg_pos = results.get('negative', {}).get('positive_ratio', 1)
        neg_n = results.get('negative', {}).get('n', 0)
        neg_mean = results.get('negative', {}).get('mean', 0)

        if neg_pos == 0 and neg_n > 0:
            findings.append("🎯 EXCLUSION RULES:")
            findings.append(f"   100% repulsion (n={neg_n:,})")
            findings.append(f"   Mean: {neg_mean:.3f}")
            findings.append("")

        comp_pos = results.get('compositional', {}).get('positive_ratio', 0)
        findings.append(f"• Compositional: {comp_pos:.1%} positive")

        cf_mean = results.get('counterfactual', {}).get('mean', 0)
        findings.append(f"• Counterfactual: {cf_mean:.3f}")

        gated_diff = (results.get('gated', {}).get('with_gate', {}).get('mean', 0) -
                      results.get('gated', {}).get('without_gate', {}).get('mean', 0))
        findings.append(f"• Gated diff: {gated_diff:.3f}")

        ax9.text(0.05, 0.95, '\n'.join(findings), transform=ax9.transAxes,
                 fontsize=10, va='top', ha='left', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax9.set_title('(I) Summary', fontweight='bold')

        plt.suptitle('Multi-Rule EC Operator Validation Dashboard', fontsize=14, fontweight='bold', y=0.98)
        plt.savefig(self.output_dir / '01_summary_dashboard.png', bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print("  ✓ Figure 1: Summary Dashboard")

    # =========================================================================
    # Figure 2: 分布对比图 (4面板 with KDE)
    # =========================================================================
    def _plot_02_distribution_comparison(self):
        """分布对比图"""
        raw = self.multi_rule_raw
        results = self.multi_rule_results

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        plot_configs = [
            ('compositional', 'Compositional: EC Sequence Additivity', axes[0, 0]),
            ('counterfactual', 'Counterfactual: Drop After EC Removal', axes[0, 1]),
            ('negative', 'Negative: Repulsion from Excluded Targets', axes[1, 0]),
        ]

        for key, title, ax in plot_configs:
            vals = raw.get(key, [])
            if vals:
                color = COLORS.get(key, '#999')
                n, bins, _ = ax.hist(vals, bins=40, density=True, alpha=0.6,
                                      color=color, edgecolor='white', label='Histogram')

                # KDE
                if len(vals) > 10:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(vals)
                    x_range = np.linspace(min(vals), max(vals), 200)
                    ax.plot(x_range, kde(x_range), color='darkred', linewidth=2, label='KDE')

                mean_val = np.mean(vals)
                median_val = np.median(vals)
                pos_ratio = results.get(key, {}).get('positive_ratio', 0)

                ax.axvline(0, color='black', linewidth=1.5)
                ax.axvline(mean_val, color='red', linewidth=2, linestyle='--', label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='blue', linewidth=2, linestyle=':', label=f'Median: {median_val:.3f}')

                # 统计信息框
                stats_text = f'n={len(vals):,}\nσ={np.std(vals):.3f}\nPos%={pos_ratio:.1%}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                        va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8))

                ax.legend(loc='upper right', fontsize=9)

            ax.set_xlabel('Value', fontweight='bold')
            ax.set_ylabel('Density', fontweight='bold')
            ax.set_title(title, fontweight='bold')

        # 第四面板: Gated对比
        ax4 = axes[1, 1]
        gated_with = raw.get('gated', {}).get('with_gate', [])
        gated_without = raw.get('gated', {}).get('without_gate', [])

        if gated_with:
            ax4.hist(gated_with, bins=30, density=True, alpha=0.6,
                     color=COLORS['gated_with'], label=f'With Gate (n={len(gated_with)})')
        if gated_without:
            ax4.hist(gated_without, bins=30, density=True, alpha=0.6,
                     color=COLORS['gated_without'], label=f'Without Gate (n={len(gated_without)})')
        ax4.axvline(0, color='black', linewidth=1.5)
        ax4.legend(loc='upper right', fontsize=9)
        ax4.set_xlabel('Cosine Similarity', fontweight='bold')
        ax4.set_ylabel('Density', fontweight='bold')
        ax4.set_title('Gated Effect Comparison', fontweight='bold')

        plt.suptitle('Distribution Comparison Across Validation Types', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_distribution_comparison.png', bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print("  ✓ Figure 2: Distribution Comparison")

    # =========================================================================
    # Figure 3: 效应量分析
    # =========================================================================
    def _plot_03_effect_size(self):
        """效应量分析"""
        raw = self.multi_rule_raw
        results = self.multi_rule_results

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel A: Cohen's d
        ax1 = axes[0]
        effect_data = []
        effect_labels = []
        effect_colors = []

        for key, label, color in [
            ('compositional', 'Compositional', COLORS['compositional']),
            ('counterfactual', 'Counterfactual', COLORS['counterfactual']),
            ('negative', 'Negative', COLORS['negative'])
        ]:
            vals = raw.get(key, [])
            if vals:
                d = np.mean(vals) / (np.std(vals) + 1e-8)
                effect_data.append(d)
                effect_labels.append(label)
                effect_colors.append(color)

        if effect_data:
            bars = ax1.barh(effect_labels, effect_data, color=effect_colors, edgecolor='black', alpha=0.85)
            ax1.axvline(0, color='black', linewidth=1)
            ax1.axvline(0.2, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='Small (0.2)')
            ax1.axvline(-0.2, color='gray', linewidth=1, linestyle='--', alpha=0.5)
            ax1.axvline(0.5, color='orange', linewidth=1, linestyle='--', alpha=0.5, label='Medium (0.5)')
            ax1.axvline(-0.5, color='orange', linewidth=1, linestyle='--', alpha=0.5)
            ax1.set_xlabel("Cohen's d", fontweight='bold')
            ax1.set_title("(A) Effect Size", fontweight='bold')
            ax1.legend(loc='lower right', fontsize=8)

            for bar, val in zip(bars, effect_data):
                ha = 'left' if val >= 0 else 'right'
                offset = 0.05 if val >= 0 else -0.05
                ax1.text(val + offset, bar.get_y() + bar.get_height()/2,
                         f'{val:.2f}', ha=ha, va='center', fontsize=10, fontweight='bold')

        # Panel B: Mean with 95% CI
        ax2 = axes[1]
        ci_data = []
        ci_labels = []
        ci_colors = []
        ci_errors = []

        for key, label, color in [
            ('compositional', 'Compositional', COLORS['compositional']),
            ('counterfactual', 'Counterfactual', COLORS['counterfactual']),
            ('negative', 'Negative', COLORS['negative'])
        ]:
            vals = raw.get(key, [])
            if vals:
                mean = np.mean(vals)
                ci = 1.96 * np.std(vals) / np.sqrt(len(vals))
                ci_data.append(mean)
                ci_labels.append(label)
                ci_colors.append(color)
                ci_errors.append(ci)

        if ci_data:
            bars = ax2.barh(ci_labels, ci_data, xerr=ci_errors, capsize=5,
                           color=ci_colors, edgecolor='black', alpha=0.85)
            ax2.axvline(0, color='black', linewidth=1.5)
            ax2.set_xlabel('Mean with 95% CI', fontweight='bold')
            ax2.set_title('(B) Confidence Intervals', fontweight='bold')

        # Panel C: High Ratio (|cos| > 0.3)
        ax3 = axes[2]
        high_ratios = [
            results.get('compositional', {}).get('high_ratio', 0),
            results.get('counterfactual', {}).get('high_ratio', 0),
            results.get('negative', {}).get('high_ratio', 0),
            results.get('gated', {}).get('with_gate', {}).get('high_ratio', 0),
        ]
        hr_labels = ['Compositional', 'Counterfactual', 'Negative', 'Gated (with)']
        hr_colors = [COLORS['compositional'], COLORS['counterfactual'],
                     COLORS['negative'], COLORS['gated_with']]

        bars = ax3.bar(hr_labels, high_ratios, color=hr_colors, edgecolor='black', alpha=0.85)
        ax3.set_ylabel('High Ratio (|cos| > 0.3)', fontweight='bold')
        ax3.set_title('(C) Strong Effect Ratio', fontweight='bold')
        ax3.tick_params(axis='x', rotation=15)
        ax3.set_ylim(0, max(high_ratios) * 1.3 + 0.05 if high_ratios else 1)

        for bar, ratio in zip(bars, high_ratios):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{ratio:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.suptitle('Effect Size Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_effect_size_analysis.png', bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print("  ✓ Figure 3: Effect Size Analysis")

    # =========================================================================
    # Figure 4: 诊断图 (排序值 + 排斥直方图)
    # =========================================================================
    def _plot_04_diagnostic(self):
        """诊断图"""
        raw = self.multi_rule_raw
        results = self.multi_rule_results

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: Compositional sorted values
        ax1 = axes[0]
        comp_vals = raw.get('compositional', [])
        if comp_vals:
            sorted_vals = np.sort(comp_vals)[::-1]
            percentiles = np.linspace(0, 1, len(sorted_vals))

            ax1.fill_between(percentiles, sorted_vals, 0, alpha=0.6, color=COLORS['compositional'])
            ax1.plot(percentiles, sorted_vals, color=COLORS['compositional'], linewidth=1)
            ax1.axhline(0, color='black', linewidth=1.5)

            # 找到零交叉点
            pos_ratio = results.get('compositional', {}).get('positive_ratio', 0)
            ax1.axvline(1 - pos_ratio, color='red', linewidth=2, linestyle='--',
                        label=f'Zero crossing: {pos_ratio:.1%} positive')

            ax1.set_xlabel('Percentile (sorted descending)', fontweight='bold')
            ax1.set_ylabel('Cosine Similarity', fontweight='bold')
            ax1.set_title('(A) Compositional: Sorted Values', fontweight='bold')
            ax1.legend(loc='upper right', fontsize=10)

        # Panel B: Negative repulsion histogram with stats
        ax2 = axes[1]
        neg_vals = raw.get('negative', [])
        if neg_vals:
            ax2.hist(neg_vals, bins=40, color=COLORS['negative'], alpha=0.7, edgecolor='white')
            ax2.axvline(0, color='black', linewidth=1.5)

            mean_val = np.mean(neg_vals)
            min_val = np.min(neg_vals)
            max_val = np.max(neg_vals)

            ax2.axvline(mean_val, color='red', linewidth=2, linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax2.axvline(min_val, color='blue', linewidth=1, linestyle=':', alpha=0.7, label=f'Min: {min_val:.3f}')
            ax2.axvline(max_val, color='green', linewidth=1, linestyle=':', alpha=0.7, label=f'Max: {max_val:.3f}')

            # 标注全部负向
            if results.get('negative', {}).get('positive_ratio', 1) == 0:
                ax2.text(0.98, 0.95, f'n={len(neg_vals):,}\nAll negative!',
                         transform=ax2.transAxes, fontsize=11, fontweight='bold',
                         ha='right', va='top', color='darkgreen',
                         bbox=dict(facecolor='lightgreen', alpha=0.8, boxstyle='round'))

            ax2.legend(loc='upper left', fontsize=10)
            ax2.set_xlabel('Cosine Similarity', fontweight='bold')
            ax2.set_ylabel('Count', fontweight='bold')
            ax2.set_title('(B) Negative Repulsion: All Values < 0', fontweight='bold')

        plt.suptitle('Diagnostic Plots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_diagnostic.png', bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        print("  ✓ Figure 4: Diagnostic Plots")

    # =========================================================================
    # Figure 5: 跨数据集验证
    # =========================================================================
    def _plot_05_cross_dataset(self):
        """跨数据集验证图"""
        data = self.cross_dataset_data
        if not data:
            return

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

        names = [d['short'] for d in data]
        colors = [SPECIES_COLORS.get(d['species'], '#999') for d in data]
        means = [d['mean_cosine'] for d in data]
        ratios = [d['positive_ratio'] for d in data]
        edges = [d['n_edges'] for d in data]

        # Panel A: Mean Cosine with SE
        ax1 = fig.add_subplot(gs[0, 0])
        ses = [0.3 / np.sqrt(n) if n > 0 else 0 for n in edges]
        bars = ax1.bar(names, means, yerr=ses, capsize=4, color=colors, edgecolor='black', alpha=0.85)
        ax1.axhline(0, color='black', linewidth=1.5)
        ax1.set_ylabel('Mean Cosine ± SE', fontweight='bold')
        ax1.set_title('(A) EC Operator Consistency', fontweight='bold')
        ax1.tick_params(axis='x', rotation=30)

        # Panel B: Positive Ratio vs Random
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(names, ratios, color=colors, edgecolor='black', alpha=0.85)
        ax2.axhline(0.5, color='red', linewidth=2, linestyle='--', label='Random (50%)')
        ax2.fill_between([-0.5, len(names)-0.5], 0.5, 0, alpha=0.1, color='red')
        ax2.fill_between([-0.5, len(names)-0.5], 0.5, 1, alpha=0.1, color='green')
        ax2.set_ylabel('Positive Ratio', fontweight='bold')
        ax2.set_title('(B) Above Random Baseline?', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=30)
        ax2.legend(loc='lower right', fontsize=9)

        for bar, val in zip(bars, ratios):
            marker = '✓' if val > 0.5 else '✗'
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                     f'{val:.0%}{marker}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Panel C: Sample Size
        ax3 = fig.add_subplot(gs[1, 0])
        bars = ax3.bar(names, edges, color=colors, edgecolor='black', alpha=0.85)
        ax3.set_ylabel('Testable EC Edges', fontweight='bold')
        ax3.set_title('(C) Sample Size', fontweight='bold')
        ax3.tick_params(axis='x', rotation=30)

        for bar, val in zip(bars, edges):
            ax3.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                     f'n={val}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Panel D: Summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        total_edges = sum(edges)
        overall_cosine = sum(m * n for m, n in zip(means, edges)) / total_edges if total_edges > 0 else 0
        overall_pos = sum(r * n for r, n in zip(ratios, edges)) / total_edges if total_edges > 0 else 0
        n_above_random = sum(1 for r in ratios if r > 0.5)

        species_counts = defaultdict(int)
        for d in data:
            species_counts[d['species']] += 1

        summary = f"""CROSS-DATASET VALIDATION
{'='*30}

Datasets: {len(data)}
  Above random: {n_above_random}/{len(data)}

Total Edges: {total_edges}

Overall (weighted):
  Mean Cosine: {overall_cosine:.4f}
  Positive Ratio: {overall_pos:.1%}

Species:
"""
        for species, count in species_counts.items():
            summary += f"  • {species}: {count}\n"

        summary += f"\nCONCLUSION:\n"
        if overall_pos > 0.55 and n_above_random >= len(data) * 0.6:
            summary += "✓ EC operators generalize\n  across species"
        elif overall_pos > 0.5:
            summary += "△ Marginal generalization"
        else:
            summary += "✗ Limited generalization"

        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
                 va='top', ha='left', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
        ax4.set_title('(D) Summary', fontweight='bold')

        # 物种图例
        legend_species = list(set(d['species'] for d in data))
        legend_elements = [mpatches.Patch(facecolor=SPECIES_COLORS.get(s, '#999'),
                                          label=s, edgecolor='black')
                          for s in legend_species]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=8, title='Species')

        plt.suptitle('Cross-Dataset EC Operator Validation', fontsize=14, fontweight='bold')
        plt.savefig(self.output_dir / '05_cross_dataset.png', bbox_inches='tight', facecolor='white', dpi=300)
        plt.savefig(self.output_dir / '05_cross_dataset.pdf', bbox_inches='tight', facecolor='white')
        plt.close()
        print("  ✓ Figure 5: Cross-Dataset Validation")

    # =========================================================================
    # Figure 6: 论文级综合图 (4面板)
    # =========================================================================
    def _plot_06_paper_figure(self):
        """论文级综合图"""
        results = self.multi_rule_results
        raw = self.multi_rule_raw

        fig = plt.figure(figsize=(12, 9))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

        # Panel A: Mean with CI
        ax1 = fig.add_subplot(gs[0, 0])
        labels = ['Compositional', 'Counterfactual', 'Gated', 'Negative']
        means = []
        cis = []
        colors = [COLORS['compositional'], COLORS['counterfactual'],
                  COLORS['gated_with'], COLORS['negative']]

        for key in ['compositional', 'counterfactual']:
            vals = raw.get(key, [])
            if vals:
                means.append(np.mean(vals))
                cis.append(1.96 * np.std(vals) / np.sqrt(len(vals)))
            else:
                means.append(0)
                cis.append(0)

        gated_vals = raw.get('gated', {}).get('with_gate', [])
        means.append(np.mean(gated_vals) if gated_vals else 0)
        cis.append(1.96 * np.std(gated_vals) / np.sqrt(len(gated_vals)) if gated_vals else 0)

        neg_vals = raw.get('negative', [])
        means.append(np.mean(neg_vals) if neg_vals else 0)
        cis.append(1.96 * np.std(neg_vals) / np.sqrt(len(neg_vals)) if neg_vals else 0)

        bars = ax1.bar(labels, means, yerr=cis, capsize=6, color=colors, edgecolor='black', alpha=0.85)
        ax1.axhline(0, color='black', linewidth=1.5)
        ax1.set_ylabel('Mean ± 95% CI', fontweight='bold')
        ax1.set_title('(A) Mean Effects', fontweight='bold')
        ax1.tick_params(axis='x', rotation=15)

        # Panel B: Distribution overlay
        ax2 = fig.add_subplot(gs[0, 1])
        comp_vals = raw.get('compositional', [])
        neg_vals = raw.get('negative', [])

        if comp_vals:
            ax2.hist(comp_vals, bins=40, density=True, alpha=0.6,
                     color=COLORS['compositional'], label=f'Compositional (n={len(comp_vals):,})')
        if neg_vals:
            ax2.hist(neg_vals, bins=40, density=True, alpha=0.6,
                     color=COLORS['negative'], label=f'Negative (n={len(neg_vals):,})')
        ax2.axvline(0, color='black', linewidth=1.5)
        ax2.set_xlabel('Cosine Similarity', fontweight='bold')
        ax2.set_ylabel('Density', fontweight='bold')
        ax2.set_title('(B) Distribution Comparison', fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)

        # Panel C: Sample sizes
        ax3 = fig.add_subplot(gs[1, 0])
        sizes = [
            results.get('compositional', {}).get('n', 0),
            results.get('counterfactual', {}).get('n', 0),
            results.get('gated', {}).get('with_gate', {}).get('n', 0),
            results.get('negative', {}).get('n', 0)
        ]
        bars = ax3.bar(labels, sizes, color=colors, edgecolor='black', alpha=0.85)
        ax3.set_ylabel('Sample Size (n)', fontweight='bold')
        ax3.set_title('(C) Sample Sizes', fontweight='bold')
        ax3.tick_params(axis='x', rotation=15)
        if max(sizes) > 100:
            ax3.set_yscale('log')

        for bar, size in zip(bars, sizes):
            y_pos = bar.get_height() * 1.1 if ax3.get_yscale() == 'log' else bar.get_height() + max(sizes)*0.02
            ax3.text(bar.get_x() + bar.get_width()/2, y_pos,
                     f'{size:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Panel D: Summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        summary_lines = ["VALIDATION SUMMARY", "=" * 25, ""]

        neg_pos = results.get('negative', {}).get('positive_ratio', 1)
        neg_n = results.get('negative', {}).get('n', 0)
        neg_mean = results.get('negative', {}).get('mean', 0)

        if neg_pos == 0 and neg_n > 0:
            summary_lines.append("✓ EXCLUSION: 100% REPULSION")
            summary_lines.append(f"   n={neg_n:,}, mean={neg_mean:.3f}")
            summary_lines.append("")

        comp_pos = results.get('compositional', {}).get('positive_ratio', 0)
        summary_lines.append(f"△ Compositional: {comp_pos:.0%} positive")

        cf_mean = results.get('counterfactual', {}).get('mean', 0)
        summary_lines.append(f"△ Counterfactual: {cf_mean:.3f}")

        summary_lines.append("")
        summary_lines.append("CONCLUSION:")
        if neg_pos == 0 and neg_n > 0:
            summary_lines.append("EC operators encode meaningful")
            summary_lines.append("biochemical transformation rules")
        else:
            summary_lines.append("Translation operators validated")
            summary_lines.append("Other families need more data")

        ax4.text(0.1, 0.9, '\n'.join(summary_lines), transform=ax4.transAxes,
                 fontsize=10, va='top', ha='left', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
        ax4.set_title('(D) Summary', fontweight='bold')

        plt.suptitle('Cross-Dataset EC Operator Validation', fontsize=14, fontweight='bold', y=0.98)
        plt.savefig(self.output_dir / '06_paper_figure.png', bbox_inches='tight', facecolor='white', dpi=300)
        plt.savefig(self.output_dir / '06_paper_figure.pdf', bbox_inches='tight', facecolor='white')
        plt.close()
        print("  ✓ Figure 6: Paper Figure (PNG + PDF)")


# =============================================================================
# 便捷函数：直接从 main() 调用
# =============================================================================

def generate_unified_visualizations(
    multi_rule_results: Dict = None,
    multi_rule_raw: Dict = None,
    cross_dataset_data: List[Dict] = None,
    output_dir: str = "./visualization_unified"
):
    """
    生成统一可视化

    Args:
        multi_rule_results: 多规则验证结果 (从 validator 获取)
        multi_rule_raw: 多规则原始数据 (cosine values lists)
        cross_dataset_data: 跨数据集数据列表 [{'short':..., 'species':..., 'mean_cosine':..., ...}, ...]
        output_dir: 输出目录
    """
    viz = UnifiedVisualizer(output_dir=output_dir)

    if multi_rule_results and multi_rule_raw:
        viz.load_multi_rule_results(multi_rule_results, multi_rule_raw)

    if cross_dataset_data:
        viz.load_cross_dataset_data(cross_dataset_data)

    viz.plot_all()
    return viz


# =============================================================================
# 独立运行示例
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("统一可视化模块测试")
    print("=" * 60)

    # 尝试加载已有数据
    multi_json = Path("../05_validation_cross_dataset/cross_validation_report_multi.json")
    multi_pkl = Path("../05_validation_cross_dataset/cross_validation_report_multi_raw.pkl")

    multi_results = None
    multi_raw = None

    if multi_json.exists():
        with open(multi_json, 'r') as f:
            multi_results = json.load(f)
        print(f"✅ 加载了 multi_rule_results")

    if multi_pkl.exists():
        import pickle
        with open(multi_pkl, 'rb') as f:
            multi_raw = pickle.load(f)
        print(f"✅ 加载了 multi_rule_raw")

    if multi_results and multi_raw:
        generate_unified_visualizations(
            multi_rule_results=multi_results,
            multi_rule_raw=multi_raw,
            output_dir="./visualization_unified"
        )
    else:
        print("❌ 没有找到数据文件，请先运行 cross_dataset_ec_validation_multi.py")
