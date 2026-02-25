"""
v5_export.py — Data export (JSON) and interactive network visualization
"""

import json
from collections import Counter
from typing import List, Dict


def export_for_visualization(
    pattern_index: Dict,
    pathways: List[Dict],
    events: List[Dict],
    output_json: str = "odor_level_patterns_weighted.json"
) -> Dict:
    """Convert pattern index to the JSON format expected by visualization scripts."""
    triplet_counts = pattern_index['triplet_counts']
    triplet_examples = pattern_index['triplet_examples']
    ec_seq_counts = pattern_index['ec_seq_counts']
    source_to_targets = pattern_index['source_to_targets']

    # 1. top triplets
    top_triplets = []
    for (src, ec_seq, tgt), weight in sorted(triplet_counts.items(), key=lambda x: -x[1])[:10000]:
        examples = triplet_examples.get((src, ec_seq, tgt), [])
        top_triplets.append({
            'source_odor': src, 'target_odor': tgt,
            'ec_sequence': list(ec_seq),
            'weighted_frequency': round(weight, 2),
            'examples': [{'src_compound': ex['src_cpd'], 'tgt_compound': ex['tgt_cpd']} for ex in examples]
        })

    # 2. top EC sequences
    top_ec_sequences = [
        {'ec_sequence': list(ec_seq), 'weighted_frequency': round(w, 2)}
        for ec_seq, w in sorted(ec_seq_counts.items(), key=lambda x: -x[1])[:500]
    ]

    # 3. source odors
    source_totals = Counter()
    for src, targets in source_to_targets.items():
        source_totals[src] = sum(targets.values())
    top_source_odors = [{'odor': o, 'total_weight': round(w, 2)} for o, w in source_totals.most_common(50)]

    # 4. target odors
    target_totals = Counter()
    for (src, ec_seq, tgt), weight in triplet_counts.items():
        target_totals[tgt] += weight
    top_target_odors = [{'odor': o, 'total_weight': round(w, 2)} for o, w in target_totals.most_common(50)]

    # 5. summary
    unique_sources = set(src for (src, _, _) in triplet_counts.keys())
    unique_targets = set(tgt for (_, _, tgt) in triplet_counts.keys())
    summary = {
        'pathways': len(pathways),
        'odor_events': len(events),
        'unique_triplets': len(triplet_counts),
        'unique_ec_sequences': len(ec_seq_counts),
        'unique_source_odors': len(unique_sources),
        'unique_target_odors': len(unique_targets),
        'expansion_ratio': round(len(events) / max(len(pathways), 1), 2)
    }

    viz_data = {
        'summary': summary,
        'top_triplets': top_triplets,
        'top_ec_sequences': top_ec_sequences,
        'top_source_odors': top_source_odors,
        'top_target_odors': top_target_odors
    }

    with open(output_json, 'w') as f:
        json.dump(viz_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 可视化数据已导出: {output_json}")
    print(f"   - {len(top_triplets)} triplets, {len(top_ec_sequences)} EC sequences")
    return viz_data


def visualize_odor_network(analyzer, output_file="odor_network.html",
                           max_nodes=500, only_odorous=False):
    """Interactive network visualization (requires pyvis)."""
    from pyvis.network import Network

    G = analyzer.reaction_graph
    if only_odorous:
        odor_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'odor_compound']
        related_reactions = set()
        for node in odor_nodes:
            related_reactions.update(G.predecessors(node))
            related_reactions.update(G.successors(node))
        G_sub = G.subgraph(set(odor_nodes) | related_reactions).copy()
    else:
        G_sub = G

    if G_sub.number_of_nodes() > max_nodes:
        degrees = dict(G_sub.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G_sub = G_sub.subgraph(top_nodes).copy()

    print(f"  可视化子图: {G_sub.number_of_nodes()} 节点, {G_sub.number_of_edges()} 边")

    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    colors = {'odor_compound': '#FF6B6B', 'intermediate': '#4ECDC4', 'reaction': '#FFE66D'}
    node_counts = {'odor_compound': 0, 'intermediate': 0, 'reaction': 0}

    for node, data in G_sub.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        node_counts[node_type] = node_counts.get(node_type, 0) + 1
        if node_type == 'odor_compound':
            label, size = node, 25
            title = f"<b>{node}</b><br>Odors: {', '.join(data.get('dominant_odors', []))}"
        elif node_type == 'reaction':
            label, size = node.replace('R_', ''), 12
            ecs = data.get('ec_numbers', [])
            title = f"<b>{node}</b><br>EC: {', '.join(ecs[:3]) if ecs else 'No EC'}"
        else:
            label, size = node, 10
            title = f"<b>{node}</b><br>Intermediate"
        net.add_node(node, label=label, color=colors.get(node_type, '#888888'),
                     size=size, title=title, font={'size': 10})

    for u, v in G_sub.edges():
        net.add_edge(u, v, arrows='to', color='#555555', width=0.5)

    net.set_options('''{
      "physics": {"forceAtlas2Based": {"gravitationalConstant": -100, "springLength": 150},
                  "solver": "forceAtlas2Based",
                  "stabilization": {"iterations": 200}},
      "interaction": {"hover": true, "tooltipDelay": 100, "navigationButtons": true}
    }''')
    net.save_graph(output_file)
    print(f"✅ 交互式网络已保存到: {output_file}")
    return net
