"""
v5_rules.py — Cartesian expansion, FOL rule extraction, query engine
"""

import json
from collections import defaultdict, Counter
from typing import List, Dict
from dataclasses import dataclass, field

from v5_config import BACKGROUND_EC, to_ec_level


# =====================================================================
# Cartesian expansion
# =====================================================================

def expand_to_odor_events(pathways: List[Dict]) -> List[Dict]:
    """Expand pathways to odor-level events (Cartesian product + weight sharing)."""
    events = []
    for pw in pathways:
        ec3_seq = []
        for ec in pw.get('ec_sequence', []):
            ec_str = str(ec).lower()
            if ec_str not in ['unknown', 'ec:unknown', '']:
                ec3 = to_ec_level(ec, level=3)
                if ec3:
                    ec3_seq.append(ec3)
        if not ec3_seq:
            continue

        source_odors = pw.get('source_odors', [])
        target_odors = pw.get('target_odors', [])
        n_source, n_target = len(source_odors), len(target_odors)
        if n_source == 0 or n_target == 0:
            continue

        weight = 1.0 / (n_source * n_target)
        for s_odor in source_odors:
            for t_odor in target_odors:
                events.append({
                    'source_odor': s_odor, 'target_odor': t_odor,
                    'ec_sequence': tuple(ec3_seq),
                    'source_compound': pw['source'], 'target_compound': pw['target'],
                    'weight': weight
                })
    return events


def build_odor_pattern_index(events: List[Dict]) -> Dict:
    """Build weighted pattern index from odor events."""
    triplet_counts = Counter()
    triplet_examples = defaultdict(list)
    pair_to_chains = defaultdict(Counter)
    source_to_targets = defaultdict(Counter)
    chain_to_pairs = defaultdict(Counter)
    ec_seq_counts = Counter()

    for event in events:
        s, t = event['source_odor'], event['target_odor']
        ec, w = event['ec_sequence'], event['weight']
        triplet_counts[(s, ec, t)] += w
        pair_to_chains[(s, t)][ec] += w
        source_to_targets[s][t] += w
        chain_to_pairs[ec][(s, t)] += w
        ec_seq_counts[ec] += w
        key = (s, ec, t)
        if len(triplet_examples[key]) < 3:
            ex = {'src_cpd': event['source_compound'], 'tgt_cpd': event['target_compound']}
            if ex not in triplet_examples[key]:
                triplet_examples[key].append(ex)

    return {
        'triplet_counts': triplet_counts,
        'triplet_examples': triplet_examples,
        'pair_to_chains': pair_to_chains,
        'source_to_targets': source_to_targets,
        'chain_to_pairs': chain_to_pairs,
        'ec_seq_counts': ec_seq_counts
    }


# =====================================================================
# FOL Rule data structure
# =====================================================================

@dataclass
class ComplexFOLRule:
    """Complex first-order logic rule."""
    rule_type: str
    quantifier: str
    head: Dict
    body: Dict
    support: float
    confidence: float
    coverage: float
    examples: List[Dict] = field(default_factory=list)

    def to_fol_string(self) -> str:
        if self.rule_type == 'necessary':
            src, tgt = self.body['source'], self.body['target']
            return f"∀P: transforms({src}, {tgt}, P) → via_ec(P, {self.head['ec']})"
        elif self.rule_type == 'sufficient':
            ec_conds = ' ∧ '.join([f"via_ec(P, {e})" for e in self.body['ec_set']])
            return f"{ec_conds} → produces(P, {self.head['target']})"
        elif self.rule_type == 'exclusion':
            src = self.body.get('source', '*')
            ec_seq = self.body['ec_sequence']
            ec_str = ' → '.join(ec_seq) if isinstance(ec_seq, (list, tuple)) else ec_seq
            return f"from_odor(P, {src}) ∧ via_ec(P, [{ec_str}]) → ¬to_odor(P, {self.head['excluded_target']})"
        elif self.rule_type == 'disjunctive_source':
            src_disj = ' ∨ '.join([f"from({s})" for s in self.body['sources']])
            ec_str = ' → '.join(self.body['ec_sequence'])
            return f"({src_disj}) ∧ via([{ec_str}]) → to({self.head['target']})"
        elif self.rule_type == 'mutual_exclusion':
            return f"via_ec(P, {self.body['ec1']}) ∧ via_ec(P, {self.body['ec2']}) → ⊥  {self.body.get('context', '')}"
        elif self.rule_type == 'conditional_necessary':
            src, tgt = self.body['source'], self.body['target']
            return f"from({src}) ∧ to({tgt}) → must_via({self.head['necessary_ec']})"
        elif self.rule_type == 'ec_class_rule':
            return f"via_ec_class(P, {self.body['ec_class']}) → {self.head['effect']}"
        return f"[{self.rule_type}] {self.body} → {self.head}"

    def to_prolog(self) -> str:
        if self.rule_type == 'necessary':
            return f"must_via(P, '{self.head['ec']}') :- transforms(P, '{self.body['source']}', '{self.body['target']}')."
        elif self.rule_type == 'sufficient':
            ec_conds = ', '.join([f"via_ec(P, '{e}')" for e in self.body['ec_set']])
            return f"produces(P, '{self.head['target']}') :- {ec_conds}."
        elif self.rule_type == 'exclusion':
            return f"impossible(P, '{self.head['excluded_target']}') :- from(P, '{self.body.get('source', '_')}'), via_seq(P, {list(self.body['ec_sequence'])})."
        elif self.rule_type == 'conditional_necessary':
            return f"required_ec('{self.body['source']}', '{self.body['target']}', '{self.head['necessary_ec']}')."
        return f"% {self.rule_type}: {self.to_fol_string()}"


# =====================================================================
# Rule extractor
# =====================================================================

class ComplexRuleExtractor:
    """Complex FOL rule extractor: necessary, sufficient, exclusion, disjunctive, mutual-exclusion, conditional."""

    def __init__(self, pattern_index: Dict,
                 min_support: float = 2.0, min_confidence: float = 0.1,
                 high_confidence: float = 0.8, exclusion_threshold: float = 0.95):
        self.index = pattern_index
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.high_confidence = high_confidence
        self.exclusion_threshold = exclusion_threshold
        self.rules: List[ComplexFOLRule] = []
        self._precompute_statistics()

    def _precompute_statistics(self):
        triplet_counts = self.index['triplet_counts']
        self.pair_ec_sets = defaultdict(list)
        self.pair_ec_weights = defaultdict(lambda: defaultdict(float))
        self.ec_to_pairs = defaultdict(lambda: defaultdict(float))
        self.ec_to_targets = defaultdict(Counter)
        self.source_ec_usage = defaultdict(Counter)
        self.ec_cooccurrence = Counter()
        self.ec_total_usage = Counter()
        self.all_targets = set()

        for (src, ec_seq, tgt), weight in triplet_counts.items():
            self.pair_ec_sets[(src, tgt)].append(ec_seq)
            self.pair_ec_weights[(src, tgt)][ec_seq] += weight
            self.ec_to_pairs[ec_seq][(src, tgt)] += weight
            self.ec_to_targets[ec_seq][tgt] += weight
            self.source_ec_usage[src][ec_seq] += weight
            for i, ec in enumerate(ec_seq):
                self.ec_total_usage[ec] += weight
                for j in range(i + 1, len(ec_seq)):
                    pair = tuple(sorted([ec_seq[i], ec_seq[j]]))
                    self.ec_cooccurrence[pair] += weight
            self.all_targets.add(tgt)

        self.target_global_counts = Counter()
        for (src, ec_seq, tgt), w in triplet_counts.items():
            self.target_global_counts[tgt] += w
        self.total_triplet_weight = sum(triplet_counts.values())
        print(f"  预计算完成: {len(self.pair_ec_sets)} 对转化, {len(self.ec_to_pairs)} 种EC序列")

    def extract_all_rules(self) -> List[ComplexFOLRule]:
        print("=" * 80)
        print("复杂一阶逻辑规则提取")
        print("=" * 80)
        self.rules = []

        for label, method in [
            ("Necessary Condition", self._extract_necessary_rules),
            ("Sufficient Condition", self._extract_sufficient_rules),
            ("Exclusion", self._extract_exclusion_rules),
            ("Disjunctive Source", self._extract_disjunctive_source_rules),
            ("Mutual Exclusion", self._extract_mutual_exclusion_rules),
            ("Conditional Necessary", self._extract_conditional_necessary_rules),
        ]:
            print(f"\n🔷 提取 {label}...")
            rules = method()
            self.rules.extend(rules)
            print(f"   ✓ {len(rules)} 条")

        self.rules.sort(key=lambda r: (r.rule_type, -r.support))
        print(f"\n{'=' * 80}\n✅ 总计 {len(self.rules)} 条规则\n{'=' * 80}")
        return self.rules

    # --- individual rule extractors ---

    def _extract_necessary_rules(self):
        rules = []
        for (src, tgt), ec_sequences in self.pair_ec_sets.items():
            if len(ec_sequences) < 2:
                continue
            common_ecs = set(ec_sequences[0])
            for seq in ec_sequences[1:]:
                common_ecs &= set(seq)
            common_ecs -= BACKGROUND_EC
            if not common_ecs:
                continue
            total_weight = sum(self.pair_ec_weights[(src, tgt)].values())
            if total_weight < self.min_support:
                continue
            for ec in common_ecs:
                rules.append(ComplexFOLRule(
                    rule_type='necessary', quantifier='forall',
                    head={'ec': ec}, body={'source': src, 'target': tgt},
                    support=total_weight, confidence=1.0,
                    coverage=len(ec_sequences) / len(self.pair_ec_sets),
                    examples=[{'paths': len(ec_sequences)}]))
        rules.sort(key=lambda r: r.support, reverse=True)
        return rules

    def _extract_sufficient_rules(self):
        rules = []
        for ec_seq, target_counts in self.ec_to_targets.items():
            total_usage = sum(target_counts.values())
            if total_usage < self.min_support:
                continue
            for target, target_weight in target_counts.items():
                confidence = target_weight / total_usage
                if confidence >= self.high_confidence:
                    ec_set = tuple(sorted(set(ec_seq)))
                    rules.append(ComplexFOLRule(
                        rule_type='sufficient', quantifier='none',
                        head={'target': target},
                        body={'ec_set': ec_set, 'ec_sequence': ec_seq},
                        support=target_weight, confidence=confidence,
                        coverage=target_weight / total_usage))
        unique_rules = {}
        for rule in rules:
            key = (rule.body['ec_set'], rule.head['target'])
            if key not in unique_rules or rule.confidence > unique_rules[key].confidence:
                unique_rules[key] = rule
        rules = list(unique_rules.values())
        rules.sort(key=lambda r: (r.confidence, r.support), reverse=True)
        return rules

    def _extract_exclusion_rules(self):
        rules = []
        min_global_weight = self.total_triplet_weight * 0.01
        common_targets = {tgt for tgt, w in self.target_global_counts.items() if w >= min_global_weight}
        for ec_seq, target_counts in self.ec_to_targets.items():
            ec_total = sum(target_counts.values())
            if ec_total < self.min_support * 5:
                continue
            source_counts = Counter()
            for (src, tgt), weight in self.ec_to_pairs[ec_seq].items():
                source_counts[src] += weight
            if not source_counts:
                continue
            top_source = source_counts.most_common(1)[0][0]
            produced_targets = set(target_counts.keys())
            for excluded_tgt in common_targets - produced_targets:
                expected_ratio = self.target_global_counts[excluded_tgt] / self.total_triplet_weight
                rules.append(ComplexFOLRule(
                    rule_type='exclusion', quantifier='forall',
                    head={'excluded_target': excluded_tgt},
                    body={'source': top_source, 'ec_sequence': ec_seq},
                    support=ec_total, confidence=1.0, coverage=expected_ratio,
                    examples=[{'global_ratio': f"{expected_ratio:.1%}"}]))
        rules.sort(key=lambda r: r.support * r.coverage, reverse=True)
        return rules

    def _extract_disjunctive_source_rules(self):
        rules = []
        ec_target_sources = defaultdict(lambda: defaultdict(float))
        for (src, ec_seq, tgt), weight in self.index['triplet_counts'].items():
            ec_target_sources[(ec_seq, tgt)][src] += weight
        for (ec_seq, tgt), source_weights in ec_target_sources.items():
            if len(source_weights) < 2:
                continue
            total_weight = sum(source_weights.values())
            if total_weight < self.min_support:
                continue
            top_sources = [s for s, _ in Counter(source_weights).most_common(5)]
            if len(top_sources) >= 2:
                rules.append(ComplexFOLRule(
                    rule_type='disjunctive_source', quantifier='exists',
                    head={'target': tgt},
                    body={'sources': top_sources, 'ec_sequence': ec_seq},
                    support=total_weight, confidence=1.0,
                    coverage=len(top_sources) / len(self.source_ec_usage)))
        rules.sort(key=lambda r: (len(r.body['sources']), r.support), reverse=True)
        return rules

    def _extract_mutual_exclusion_rules(self):
        rules = []
        total_paths = sum(self.index['ec_seq_counts'].values())
        ec_list = [ec for ec, count in self.ec_total_usage.items() if count >= self.min_support]
        for i, ec1 in enumerate(ec_list):
            for ec2 in ec_list[i + 1:]:
                pair = tuple(sorted([ec1, ec2]))
                actual = self.ec_cooccurrence.get(pair, 0)
                p1 = self.ec_total_usage[ec1] / total_paths
                p2 = self.ec_total_usage[ec2] / total_paths
                expected = p1 * p2 * total_paths
                if expected > 5:
                    ratio = actual / expected
                    if ratio < 0.1:
                        rules.append(ComplexFOLRule(
                            rule_type='mutual_exclusion', quantifier='forall',
                            head={'contradiction': True},
                            body={'ec1': ec1, 'ec2': ec2, 'context': f"(ratio={ratio:.2f})"},
                            support=expected, confidence=1 - ratio, coverage=ratio))
        rules.sort(key=lambda r: r.confidence, reverse=True)
        return rules

    def _extract_conditional_necessary_rules(self):
        rules = []
        for (src, tgt), ec_weights in self.pair_ec_weights.items():
            total_weight = sum(ec_weights.values())
            if total_weight < self.min_support:
                continue
            ec_frequency = Counter()
            for ec_seq, weight in ec_weights.items():
                for ec in ec_seq:
                    ec_frequency[ec] += weight
            for ec, ec_weight in ec_frequency.items():
                if ec in BACKGROUND_EC:
                    continue
                coverage = ec_weight / total_weight
                if coverage >= 0.8:
                    rules.append(ComplexFOLRule(
                        rule_type='conditional_necessary', quantifier='forall',
                        head={'necessary_ec': ec},
                        body={'source': src, 'target': tgt},
                        support=total_weight, confidence=coverage, coverage=coverage))
        rules.sort(key=lambda r: (r.confidence, r.support), reverse=True)
        return rules

    # --- query / export helpers ---

    def get_rules_by_type(self, rule_type: str):
        return [r for r in self.rules if r.rule_type == rule_type]

    def get_rules_for_odor(self, odor: str, role: str = 'any'):
        results = []
        for rule in self.rules:
            match = False
            if role in ['source', 'any']:
                if rule.body.get('source') == odor or odor in rule.body.get('sources', []):
                    match = True
            if role in ['target', 'any']:
                if rule.head.get('target') == odor or rule.head.get('excluded_target') == odor:
                    match = True
            if match:
                results.append(rule)
        return results

    def get_rules_for_ec(self, ec: str):
        results = []
        for rule in self.rules:
            if rule.head.get('ec') == ec or rule.head.get('necessary_ec') == ec:
                results.append(rule); continue
            if ec in rule.body.get('ec_set', ()):
                results.append(rule); continue
            if ec in rule.body.get('ec_sequence', ()):
                results.append(rule); continue
            if rule.body.get('ec1') == ec or rule.body.get('ec2') == ec:
                results.append(rule); continue
        return results

    def print_rules_summary(self):
        print("\n" + "=" * 80 + "\n规则摘要\n" + "=" * 80)
        type_counts = Counter(r.rule_type for r in self.rules)
        for rt, count in type_counts.most_common():
            print(f"  {rt}: {count} 条")
        print(f"\n  总计: {len(self.rules)} 条规则")

    def print_top_rules(self, n: int = 10, by_type: bool = True):
        print("\n" + "=" * 100 + f"\nTop {n} 复杂一阶逻辑规则\n" + "=" * 100)
        if by_type:
            type_counts = Counter(r.rule_type for r in self.rules)
            for rule_type in type_counts.keys():
                type_rules = self.get_rules_by_type(rule_type)[:n]
                if not type_rules:
                    continue
                print(f"\n{'─' * 80}\n📌 {rule_type.upper()} ({len(self.get_rules_by_type(rule_type))} 条)\n{'─' * 80}")
                for i, rule in enumerate(type_rules, 1):
                    print(f"\n  #{i} [support={rule.support:.1f}, conf={rule.confidence:.2f}]")
                    print(f"     {rule.to_fol_string()}")
        else:
            for i, rule in enumerate(self.rules[:n], 1):
                print(f"\n#{i} [{rule.rule_type}] (support={rule.support:.1f}, conf={rule.confidence:.2f})")
                print(f"   {rule.to_fol_string()}")

    def export_rules(self, filepath: str = 'complex_fol_rules.json') -> Dict:
        rules_by_type = defaultdict(list)
        for rule in self.rules:
            rules_by_type[rule.rule_type].append(rule)
        export_data = {
            'summary': {
                'total_rules': len(self.rules),
                'by_type': {k: len(v) for k, v in rules_by_type.items()},
                'parameters': {'min_support': self.min_support,
                               'min_confidence': self.min_confidence,
                               'high_confidence': self.high_confidence}
            },
            'rules': {},
            'prolog_clauses': []
        }
        for rule_type, type_rules in rules_by_type.items():
            export_data['rules'][rule_type] = [
                {
                    'id': i, 'fol_string': r.to_fol_string(), 'prolog': r.to_prolog(),
                    'head': r.head,
                    'body': {k: list(v) if isinstance(v, tuple) else v for k, v in r.body.items()},
                    'support': round(r.support, 2), 'confidence': round(r.confidence, 3),
                    'coverage': round(r.coverage, 3)
                }
                for i, r in enumerate(type_rules)
            ]
        export_data['prolog_clauses'] = [r.to_prolog() for r in self.rules[:300]]
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 规则已导出到: {filepath}")
        return export_data


# =====================================================================
# Query engine
# =====================================================================

class OdorRuleQueryEngine:
    """Odor rule query engine."""

    def __init__(self, pattern_index: Dict, rule_extractor: ComplexRuleExtractor):
        self.index = pattern_index
        self.extractor = rule_extractor

    def query_transformation(self, source_odor: str, top_k: int = 10):
        results = []
        if source_odor not in self.index['source_to_targets']:
            return []
        for target_odor, total_weight in self.index['source_to_targets'][source_odor].most_common(top_k * 2):
            chains = self.index['pair_to_chains'].get((source_odor, target_odor), {})
            if not chains:
                continue
            best_chain, chain_weight = chains.most_common(1)[0]
            results.append({
                'source': source_odor, 'target': target_odor,
                'ec_chain': list(best_chain), 'score': chain_weight,
                'rule': f"{source_odor} --[{' → '.join(best_chain)}]--> {target_odor}"
            })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def query_ec_chain(self, source_odor: str, target_odor: str, top_k: int = 5):
        chains = self.index['pair_to_chains'].get((source_odor, target_odor), {})
        return [{'ec_chain': list(ec), 'score': w,
                 'rule': f"{source_odor} --[{' → '.join(ec)}]--> {target_odor}"}
                for ec, w in chains.most_common(top_k)]

    def query_necessary_ec(self, source_odor: str, target_odor: str):
        return [r.head['necessary_ec']
                for r in self.extractor.get_rules_by_type('conditional_necessary')
                if r.body.get('source') == source_odor and r.body.get('target') == target_odor]

    def query_excluded_targets(self, source_odor: str, ec_sequence: tuple):
        return [r.head['excluded_target']
                for r in self.extractor.get_rules_by_type('exclusion')
                if r.body.get('source') == source_odor and r.body.get('ec_sequence') == ec_sequence]

    def demo_queries(self):
        print("\n" + "=" * 80 + "\n规则查询演示\n" + "=" * 80)

        print("\n📌 任务A: 给定 'mint' 气味，预测可能的转化")
        for r in self.query_transformation('mint', top_k=5):
            print(f"  {r['rule']}  (score={r['score']:.1f})")

        print("\n📌 任务B: 给定 'phenolic' → 'woody'，推断EC链")
        for r in self.query_ec_chain('phenolic', 'woody', top_k=5):
            print(f"  {r['rule']}  (score={r['score']:.1f})")

        print("\n📌 任务C: 查询 'mint' → 'woody' 的必要EC")
        necessary = self.query_necessary_ec('mint', 'woody')
        print(f"  必要EC: {', '.join(necessary)}" if necessary else "  未找到必要EC规则")

        print("\n📌 查询与 'mint' 相关的复杂规则")
        for r in self.extractor.get_rules_for_odor('mint')[:5]:
            print(f"  [{r.rule_type}] {r.to_fol_string()}")
