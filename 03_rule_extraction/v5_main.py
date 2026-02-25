#!/usr/bin/env python3
"""
v5_main.py — Orchestrator with per-step CLI selection

Usage:
    python v5_main.py                  # run all steps 1-7
    python v5_main.py --step 1         # only data generation
    python v5_main.py --step 2,3       # expansion + rule extraction
    python v5_main.py --step 5         # only visualization (uses cached data)
    python v5_main.py --step 5 --hq    # visualization at 300 DPI (publication)
    python v5_main.py --list-steps     # show available steps

Steps:
    1  Data generation (KEGG download + network)
    2  Cartesian expansion
    3  Rule extraction
    4  Export viz data (JSON)
    5  Generate charts (Figures 1-11)
    6  Query demo
    7  Interactive network (requires pyvis)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

# Ensure local modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


STEP_LABELS = {
    1: "Data generation (KEGG + network)",
    2: "Cartesian expansion",
    3: "FOL rule extraction",
    4: "Export viz JSON",
    5: "Generate charts (Figures 1-11)",
    6: "Query demo",
    7: "Interactive network (pyvis)",
}

# Pickle file for passing intermediate data between steps
_INTER_PKL = "v5_intermediate.pkl"


def _save_intermediate(**kwargs):
    existing = {}
    if Path(_INTER_PKL).exists():
        with open(_INTER_PKL, 'rb') as f:
            existing = pickle.load(f)
    existing.update(kwargs)
    with open(_INTER_PKL, 'wb') as f:
        pickle.dump(existing, f)


def _load_intermediate(*keys):
    if not Path(_INTER_PKL).exists():
        return None
    with open(_INTER_PKL, 'rb') as f:
        data = pickle.load(f)
    if keys:
        return {k: data.get(k) for k in keys}
    return data


def step_1():
    """Data generation"""
    from v5_data import OdorPathwayAnalyzer
    print("\n" + "=" * 80 + "\nSTEP 1: 数据生成\n" + "=" * 80)
    analyzer = OdorPathwayAnalyzer()
    if not analyzer.load_tgsc_data("../02_kegg_mapping/tgsc_to_kegg.csv"):
        print("❌ 无法加载TGSC数据"); return
    analyzer.download_kegg_data()
    if not analyzer.build_odor_network(max_depth=2, use_main_pairs=True):
        print("❌ 无法构建网络"); return
    pathways = analyzer.find_odor_pathways(max_length=5, max_paths_per_pair=2, use_cache=True)
    if not pathways:
        print("❌ 未找到pathways"); return
    print(f"\n✅ 获得 {len(pathways):,} 条 pathways")
    _save_intermediate(analyzer=analyzer, pathways=pathways)


def step_2():
    """Cartesian expansion"""
    from v5_rules import expand_to_odor_events, build_odor_pattern_index
    print("\n" + "=" * 80 + "\nSTEP 2: 笛卡尔展开\n" + "=" * 80)
    inter = _load_intermediate('pathways')
    if not inter or not inter.get('pathways'):
        print("❌ 需要先运行 step 1"); return
    events = expand_to_odor_events(inter['pathways'])
    print(f"展开为 {len(events):,} 个气味级事件")
    pattern_index = build_odor_pattern_index(events)
    print(f"唯一三元组: {len(pattern_index['triplet_counts']):,}")
    print(f"唯一EC序列: {len(pattern_index['ec_seq_counts']):,}")
    _save_intermediate(events=events, pattern_index=pattern_index)


def step_3():
    """Rule extraction"""
    from v5_rules import ComplexRuleExtractor
    print("\n" + "=" * 80 + "\nSTEP 3: 复杂一阶逻辑规则提取\n" + "=" * 80)
    inter = _load_intermediate('pattern_index')
    if not inter or not inter.get('pattern_index'):
        print("❌ 需要先运行 step 2"); return
    extractor = ComplexRuleExtractor(
        inter['pattern_index'],
        min_support=2.0, min_confidence=0.1,
        high_confidence=0.8, exclusion_threshold=0.95)
    extractor.extract_all_rules()
    extractor.print_rules_summary()
    extractor.print_top_rules(n=5, by_type=True)
    extractor.export_rules('complex_fol_rules.json')
    _save_intermediate(extractor=extractor)


def step_4():
    """Export viz data"""
    from v5_export import export_for_visualization
    print("\n" + "=" * 80 + "\nSTEP 4: 导出可视化数据\n" + "=" * 80)
    inter = _load_intermediate('pattern_index', 'pathways', 'events')
    if not inter or not inter.get('pattern_index'):
        print("❌ 需要先运行 step 2"); return
    viz_data = export_for_visualization(
        inter['pattern_index'], inter['pathways'], inter['events'],
        output_json="odor_level_patterns_weighted.json")
    _save_intermediate(viz_data=viz_data)


def step_5(hq=False):
    """Generate charts"""
    from v5_plots import run_all_plots, set_dpi
    print("\n" + "=" * 80 + "\nSTEP 5: 生成可视化图表\n" + "=" * 80)

    if hq:
        set_dpi(300)
        print("  📸 Publication quality (300 DPI)")
    else:
        print("  ⚡ Draft quality (150 DPI) — use --hq for 300 DPI")

    # Try intermediate cache first, then JSON file
    inter = _load_intermediate('viz_data')
    viz_data = inter.get('viz_data') if inter else None

    if not viz_data:
        json_path = Path("odor_level_patterns_weighted.json")
        if json_path.exists():
            print(f"  📂 Loading from {json_path}")
            with open(json_path, 'r') as f:
                viz_data = json.load(f)
        else:
            print("❌ 需要先运行 step 4 或确保 odor_level_patterns_weighted.json 存在"); return

    run_all_plots(viz_data, "./visualization_charts")


def step_6():
    """Query demo"""
    from v5_rules import OdorRuleQueryEngine
    print("\n" + "=" * 80 + "\nSTEP 6: 规则查询演示\n" + "=" * 80)
    inter = _load_intermediate('pattern_index', 'extractor')
    if not inter or not inter.get('extractor'):
        print("❌ 需要先运行 step 3"); return
    engine = OdorRuleQueryEngine(inter['pattern_index'], inter['extractor'])
    engine.demo_queries()


def step_7():
    """Interactive network"""
    from v5_export import visualize_odor_network
    print("\n" + "=" * 80 + "\nSTEP 7: 交互式网络可视化\n" + "=" * 80)
    inter = _load_intermediate('analyzer')
    if not inter or not inter.get('analyzer'):
        print("❌ 需要先运行 step 1"); return
    try:
        visualize_odor_network(inter['analyzer'], output_file="odor_network_interactive.html",
                               max_nodes=2000, only_odorous=False)
    except ImportError:
        print("❌ 请先安装 pyvis: pip install pyvis")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")


STEP_FUNCS = {1: step_1, 2: step_2, 3: step_3, 4: step_4, 5: step_5, 6: step_6, 7: step_7}


def main():
    parser = argparse.ArgumentParser(
        description="气味规则提取与可视化系统 v5 (模块化版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--step', type=str, default=None,
                        help='要运行的步骤 (逗号分隔), e.g. "1,2,3" or "5"')
    parser.add_argument('--hq', action='store_true',
                        help='Publication quality (300 DPI) for step 5')
    parser.add_argument('--list-steps', action='store_true',
                        help='列出所有可用步骤')
    args = parser.parse_args()

    if args.list_steps:
        print("可用步骤:")
        for num, label in STEP_LABELS.items():
            print(f"  {num}: {label}")
        return

    print("=" * 80)
    print("气味规则提取与可视化系统 v5 (模块化版)")
    print("=" * 80)

    if args.step:
        steps = [int(s.strip()) for s in args.step.split(',')]
    else:
        steps = list(range(1, 8))

    for s in steps:
        if s not in STEP_FUNCS:
            print(f"❌ 未知步骤: {s}")
            continue
        if s == 5:
            step_5(hq=args.hq)
        else:
            STEP_FUNCS[s]()

    print("\n" + "=" * 80)
    print("🎉 完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
