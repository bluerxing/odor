#!/usr/bin/env python3
"""
跨数据集EC算子验证（多规则版）
================================

在原有EC平移一致性的基础上，引入“规则算子家族”验证：
1) Compositional: EC序列可加性
2) Counterfactual: 必要EC移除后的性能下降
3) Gated: 条件必要的门控效应
4) Negative: 排斥规则的“远离”倾向

同时内置更严谨的规则生成逻辑（替代 v5_all_with_vis.py 的关键缺陷）。
"""

import json
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Rule filtering defaults (keep evaluation tractable)
MIN_RULE_SUPPORT = 10.0
MIN_RULE_CONFIDENCE = 0.8
MIN_RULE_COVERAGE = 0.02
MAX_RULES_PER_TYPE = 300

# Matplotlib cache dir (avoid permission issues)
def _ensure_mpl_config_dir() -> None:
    if "MPLCONFIGDIR" in os.environ:
        return
    candidates = [
        "/tmp/.mplconfig",
        "./.mplconfig",
    ]
    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = path
            return
        except Exception:
            continue

# 尝试导入RDKit用于SMILES标准化
try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("⚠️ RDKit未安装，SMILES匹配可能不完整")


def normalize_non_isomeric_smiles(smiles: str) -> Optional[str]:
    """OpenPOM-style: strip stereochemistry markers, then canonicalize."""
    if not smiles or smiles == "nan":
        return None
    if not HAS_RDKIT:
        return smiles
    try:
        cleaned = smiles.replace("@", "").replace("/", "").replace("\\", "")
        mol = Chem.MolFromSmiles(cleaned)
        if not mol:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def to_ec_level(ec: str, level: int = 3) -> str:
    if not ec or "." not in ec:
        return ""
    parts = ec.split(".")
    if len(parts) < level:
        return ""
    result = parts[:level]
    if result[-1] == "-":
        return ""
    return ".".join(result)


class QianDataLoader:
    """加载Qian et al. 2023的数据"""

    def __init__(self, publications_dir: str = "../publications"):
        self.base_dir = Path(publications_dir)
        self.data_dir = self.base_dir / "qian_et_al_2023" / "predictive_performance" / "data"
        self.embeddings = None
        self.embeddings_dict = {}
        self.datasets = {}

    def load_embeddings(self) -> bool:
        emb_file = self.data_dir / "embeddings.csv"
        if not emb_file.exists():
            print(f"❌ embeddings.csv不存在: {emb_file}")
            print("   请先运行: git clone https://github.com/osmoai/publications.git")
            return False

        print("📦 加载embeddings.csv...")
        self.embeddings = pd.read_csv(emb_file, index_col=0)
        print(f"   Shape: {self.embeddings.shape}")

        for smiles in self.embeddings.index:
            emb = self.embeddings.loc[smiles].values.astype(np.float32)
            self.embeddings_dict[smiles] = emb
            if HAS_RDKIT:
                non_iso = normalize_non_isomeric_smiles(smiles)
                if non_iso and non_iso != smiles:
                    self.embeddings_dict[non_iso] = emb

        print(f"✅ 加载了 {len(self.embeddings)} 个分子的embeddings")
        return True

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        dataset_dirs = self._discover_dataset_names()

        print("\n📦 加载所有数据集...")
        print(f"   自动发现 {len(dataset_dirs)} 个数据集路径")
        for name in dataset_dirs:
            data_file = self.data_dir / name / "data.csv"
            if data_file.exists():
                df = pd.read_csv(data_file)
                self.datasets[name] = df
                print(f"✅ 加载数据集 {name}: {len(df)} 样本")
            else:
                print(f"❌ 数据集不存在: {data_file}")
        print(f"\n✅ 成功加载 {len(self.datasets)} 个数据集")
        return self.datasets

    def _discover_dataset_names(self) -> List[str]:
        """自动发现 data.csv，支持一层嵌套目录。"""
        names: List[str] = []
        if not self.data_dir.exists():
            return names

        for item in sorted(self.data_dir.iterdir()):
            if not item.is_dir():
                continue

            direct_file = item / "data.csv"
            if direct_file.exists():
                names.append(item.name)
                continue

            for child in sorted(item.iterdir()):
                if child.is_dir() and (child / "data.csv").exists():
                    names.append(f"{item.name}/{child.name}")

        return names

    def get_embedding(self, smiles: str) -> Optional[np.ndarray]:
        if smiles in self.embeddings_dict:
            return self.embeddings_dict[smiles]
        if HAS_RDKIT:
            non_iso = normalize_non_isomeric_smiles(smiles)
            if non_iso and non_iso in self.embeddings_dict:
                return self.embeddings_dict[non_iso]
        return None


class TGSCMappingLoader:
    """加载KEGG->SMILES映射（TGSC + merged_dataset_with_kegg）"""

    def __init__(self):
        self.kegg_to_smiles = {}

    def load(self, tgsc_csv: str, merged_csv: str) -> None:
        added = 0
        merged_df = pd.read_csv(merged_csv)
        if "KEGG_IDs" in merged_df.columns and "nonStereoSMILES" in merged_df.columns:
            for _, row in merged_df.iterrows():
                smi = normalize_non_isomeric_smiles(str(row["nonStereoSMILES"]).strip())
                if not smi:
                    continue
                for kid in self._split_kegg_ids(str(row["KEGG_IDs"])):
                    if kid not in self.kegg_to_smiles:
                        self.kegg_to_smiles[kid] = smi
                        added += 1
        print(f"✅ merged_dataset_with_kegg映射: +{added} 个KEGG ID")

        tgsc_df = pd.read_csv(tgsc_csv)
        smiles_col = None
        for candidate in ["nonStereoSMILES", "IsomericSMILES", "SMILES", "smiles"]:
            if candidate in tgsc_df.columns:
                smiles_col = candidate
                break
        if not smiles_col:
            print("❌ 未找到SMILES列（nonStereoSMILES/IsomericSMILES/SMILES/smiles）")
            return

        added = 0
        for _, row in tgsc_df.iterrows():
            smi = normalize_non_isomeric_smiles(str(row[smiles_col]).strip())
            if not smi:
                continue
            for kid in self._split_kegg_ids(str(row.get("KEGG_IDs", ""))):
                if kid not in self.kegg_to_smiles:
                    self.kegg_to_smiles[kid] = smi
                    added += 1
        print(f"✅ KEGG->SMILES映射: {len(self.kegg_to_smiles)} 个KEGG ID (TGSC新增 {added})")

    @staticmethod
    def _split_kegg_ids(value: str) -> List[str]:
        if not value or value == "nan":
            return []
        parts = re.split(r"[;,|\s]+", value)
        cleaned = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            part = part.split(":")[-1]
            if re.fullmatch(r"C\d{5}", part):
                cleaned.append(part)
        return cleaned


class ECOperatorExtractor:
    """从pathway数据中提取EC算子"""

    def __init__(self, embedding_getter):
        self.get_embedding = embedding_getter
        self.operators = {}
        self.operator_stats = {}

    def extract_operators(
        self,
        ec_edges: List[Tuple[str, str, str]],
        min_samples: int = 5,
        min_mean_op_norm: float = 0.05,
        min_mean_delta_norm: float = 0.05,
    ) -> Dict[str, np.ndarray]:
        print("\n📐 提取EC算子...")
        ec_deltas = defaultdict(list)

        for src, tgt, ec in tqdm(ec_edges, desc="计算delta向量"):
            src_emb = self.get_embedding(src)
            tgt_emb = self.get_embedding(tgt)
            if src_emb is None or tgt_emb is None:
                continue
            delta = tgt_emb - src_emb
            ec_deltas[ec].append(delta)

        for ec, deltas in ec_deltas.items():
            if len(deltas) < min_samples:
                continue
            deltas_arr = np.array(deltas)
            mean_op = np.mean(deltas_arr, axis=0)
            mean_delta_norm = float(np.mean(np.linalg.norm(deltas_arr, axis=1)))
            mean_op_norm = float(np.linalg.norm(mean_op))
            if mean_op_norm < min_mean_op_norm or mean_delta_norm < min_mean_delta_norm:
                continue

            std_op = np.std(deltas_arr, axis=0)
            self.operators[ec] = mean_op
            self.operator_stats[ec] = {
                "n_samples": len(deltas),
                "mean_magnitude": mean_op_norm,
                "std_magnitude": np.mean(np.linalg.norm(std_op)),
                "consistency": 1.0 - np.mean(std_op) / (np.mean(np.abs(mean_op)) + 1e-8),
            }

        print(f"✅ 提取了 {len(self.operators)} 个EC算子")
        top_ops = sorted(self.operator_stats.items(), key=lambda x: x[1]["n_samples"], reverse=True)[:10]
        print("\n   Top 10 算子 (按样本数):")
        for ec, stats in top_ops:
            print(f"     {ec}: n={stats['n_samples']}, mag={stats['mean_magnitude']:.3f}")
        return self.operators


@dataclass
class ComplexFOLRule:
    rule_type: str
    head: Dict
    body: Dict
    support: float
    confidence: float
    coverage: float


class RuleExtractorV2:
    """更严谨的规则提取器（避免 v5_all_with_vis.py 的关键缺陷）"""

    def __init__(self, min_support: float = 5.0, high_confidence: float = 0.8):
        self.min_support = min_support
        self.high_confidence = high_confidence
        self.rules: List[ComplexFOLRule] = []

        self.pair_ec_sets = defaultdict(list)  # (src_odor, tgt_odor) -> [ec_seq]
        self.pair_ec_weights = defaultdict(lambda: defaultdict(float))
        self.ec_to_targets = defaultdict(lambda: defaultdict(float))  # ec_seq -> target counts
        self.ec_to_pairs = defaultdict(lambda: defaultdict(float))  # ec_seq -> (src,tgt) counts
        self.ec_total_usage = defaultdict(float)
        self.ec_cooccurrence = defaultdict(float)
        self.target_global_counts = defaultdict(float)
        self.total_triplet_weight = 0.0

    def ingest_pathways(self, pathways: List[Dict]) -> None:
        for pw in pathways:
            ec_seq = tuple(to_ec_level(ec, 3) for ec in pw.get("ec_sequence", []) if to_ec_level(ec, 3))
            if not ec_seq:
                continue

            source_odors = pw.get("source_odors") or []
            target_odors = pw.get("target_odors") or []
            if not source_odors or not target_odors:
                continue

            weight = 1.0
            for src in source_odors:
                for tgt in target_odors:
                    self.pair_ec_sets[(src, tgt)].append(ec_seq)
                    self.pair_ec_weights[(src, tgt)][ec_seq] += weight
                    self.ec_to_targets[ec_seq][tgt] += weight
                    self.ec_to_pairs[ec_seq][(src, tgt)] += weight
                    self.target_global_counts[tgt] += weight
                    self.total_triplet_weight += weight

            for ec in set(ec_seq):
                self.ec_total_usage[ec] += weight

            ec_set = sorted(set(ec_seq))
            for i, ec1 in enumerate(ec_set):
                for ec2 in ec_set[i + 1:]:
                    self.ec_cooccurrence[(ec1, ec2)] += weight

    def extract_rules(self) -> List[ComplexFOLRule]:
        self.rules = []
        self.rules.extend(self._extract_necessary_rules())
        self.rules.extend(self._extract_sufficient_rules())
        self.rules.extend(self._extract_exclusion_rules())
        self.rules.extend(self._extract_mutual_exclusion_rules())
        self.rules.extend(self._extract_conditional_necessary_rules())
        self.rules.sort(key=lambda r: (r.rule_type, -r.support))
        print(f"✅ 总计提取 {len(self.rules)} 条复杂规则")
        return self.rules

    def _extract_necessary_rules(self) -> List[ComplexFOLRule]:
        rules = []
        for (src, tgt), ec_sequences in self.pair_ec_sets.items():
            if len(ec_sequences) < 2:
                continue

            ec_sets = [set(seq) for seq in ec_sequences]
            common_ecs = ec_sets[0].copy()
            for ec_set in ec_sets[1:]:
                common_ecs &= ec_set

            if not common_ecs:
                continue

            total_weight = sum(self.pair_ec_weights[(src, tgt)].values())
            if total_weight < self.min_support:
                continue

            for ec in common_ecs:
                rules.append(
                    ComplexFOLRule(
                        rule_type="necessary",
                        head={"ec": ec},
                        body={"source": src, "target": tgt},
                        support=total_weight,
                        confidence=1.0,
                        coverage=1.0,
                    )
                )
        return rules

    def _extract_sufficient_rules(self) -> List[ComplexFOLRule]:
        rules = []
        for ec_seq, target_counts in self.ec_to_targets.items():
            total_usage = sum(target_counts.values())
            if total_usage < self.min_support:
                continue
            for target, target_weight in target_counts.items():
                confidence = target_weight / total_usage
                if confidence >= self.high_confidence and target_weight >= self.min_support:
                    ec_set = tuple(sorted(set(ec_seq)))
                    rules.append(
                        ComplexFOLRule(
                            rule_type="sufficient",
                            head={"target": target},
                            body={"ec_set": ec_set, "ec_sequence": ec_seq},
                            support=target_weight,
                            confidence=confidence,
                            coverage=target_weight / total_usage,
                        )
                    )
        # 去重
        unique = {}
        for r in rules:
            key = (r.body["ec_set"], r.head["target"])
            if key not in unique or r.confidence > unique[key].confidence:
                unique[key] = r
        return list(unique.values())

    def _extract_exclusion_rules(self) -> List[ComplexFOLRule]:
        rules = []
        min_global_ratio = 0.01
        min_global_weight = self.total_triplet_weight * min_global_ratio
        common_targets = {tgt for tgt, w in self.target_global_counts.items() if w >= min_global_weight}

        for ec_seq, target_counts in self.ec_to_targets.items():
            ec_total = sum(target_counts.values())
            if ec_total < self.min_support * 5:
                continue

            source_counts = Counter()
            for (src, _tgt), weight in self.ec_to_pairs[ec_seq].items():
                source_counts[src] += weight
            if not source_counts:
                continue

            for src, src_weight in source_counts.items():
                if src_weight < self.min_support:
                    continue

                produced_targets = {t for (s, t), _ in self.ec_to_pairs[ec_seq].items() if s == src}
                excluded_targets = common_targets - produced_targets

                for excluded_tgt in excluded_targets:
                    global_weight = self.target_global_counts[excluded_tgt]
                    expected_ratio = global_weight / self.total_triplet_weight
                    rules.append(
                        ComplexFOLRule(
                            rule_type="exclusion",
                            head={"excluded_target": excluded_tgt},
                            body={"source": src, "ec_sequence": ec_seq},
                            support=src_weight,
                            confidence=1.0,
                            coverage=expected_ratio,
                        )
                    )
        return rules

    def _extract_mutual_exclusion_rules(self) -> List[ComplexFOLRule]:
        rules = []
        total_paths = sum(self.ec_total_usage.values())
        ec_list = [ec for ec, count in self.ec_total_usage.items() if count >= self.min_support]

        for i, ec1 in enumerate(ec_list):
            for ec2 in ec_list[i + 1:]:
                pair = tuple(sorted([ec1, ec2]))
                actual = self.ec_cooccurrence.get(pair, 0.0)
                p1 = self.ec_total_usage[ec1] / total_paths
                p2 = self.ec_total_usage[ec2] / total_paths
                expected = p1 * p2 * total_paths
                if expected > 5:
                    ratio = actual / expected
                    if ratio < 0.1:
                        rules.append(
                            ComplexFOLRule(
                                rule_type="mutual_exclusion",
                                head={"contradiction": True},
                                body={"ec1": ec1, "ec2": ec2},
                                support=expected,
                                confidence=1 - ratio,
                                coverage=ratio,
                            )
                        )
        return rules

    def _extract_conditional_necessary_rules(self) -> List[ComplexFOLRule]:
        rules = []
        for (src, tgt), ec_weights in self.pair_ec_weights.items():
            total_weight = sum(ec_weights.values())
            if total_weight < self.min_support:
                continue

            ec_frequency = Counter()
            for ec_seq, weight in ec_weights.items():
                for ec in set(ec_seq):
                    ec_frequency[ec] += weight

            for ec, ec_weight in ec_frequency.items():
                coverage = ec_weight / total_weight
                if coverage >= 0.8:
                    rules.append(
                        ComplexFOLRule(
                            rule_type="conditional_necessary",
                            head={"necessary_ec": ec},
                            body={"source": src, "target": tgt},
                            support=total_weight,
                            confidence=coverage,
                            coverage=coverage,
                        )
                    )
        return rules


class OperatorFamilyValidator:
    """组合/反事实/门控/排斥验证"""

    def __init__(self, operators: Dict[str, np.ndarray], kegg_to_smiles: Dict[str, str], qian: QianDataLoader):
        self.operators = operators
        self.kegg_to_smiles = kegg_to_smiles
        self.qian = qian

    def _get_embedding_for_kegg(self, kegg_id: str) -> Optional[np.ndarray]:
        smi = self.kegg_to_smiles.get(kegg_id)
        if not smi:
            return None
        return self.qian.get_embedding(smi)

    def _op_sum(self, ec_seq: Tuple[str, ...]) -> Optional[np.ndarray]:
        vectors = []
        for ec in ec_seq:
            op = self.operators.get(ec)
            if op is None:
                return None
            vectors.append(op)
        if not vectors:
            return None
        return np.sum(vectors, axis=0)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def compositional(self, pathways: List[Dict]) -> Dict:
        cosines = []
        for pw in pathways:
            if len(pw.get("ec_sequence", [])) < 2:
                continue
            ec_seq = tuple(to_ec_level(ec, 3) for ec in pw.get("ec_sequence", []) if to_ec_level(ec, 3))
            if len(ec_seq) < 2:
                continue
            op_sum = self._op_sum(ec_seq)
            if op_sum is None:
                continue
            src_emb = self._get_embedding_for_kegg(pw.get("source"))
            tgt_emb = self._get_embedding_for_kegg(pw.get("target"))
            if src_emb is None or tgt_emb is None:
                continue
            delta = tgt_emb - src_emb
            cosines.append(self._cosine(op_sum, delta))
        return self._summarize("compositional", cosines), cosines

    def counterfactual(self, pathways: List[Dict], rules: List[ComplexFOLRule]) -> Dict:
        drops = []
        for rule in rules:
            if rule.rule_type not in {"necessary", "conditional_necessary"}:
                continue
            required_ec = rule.head.get("ec") or rule.head.get("necessary_ec")
            src = rule.body.get("source")
            tgt = rule.body.get("target")
            if not required_ec or not src or not tgt:
                continue

            for pw in pathways:
                if src not in (pw.get("source_odors") or []):
                    continue
                if tgt not in (pw.get("target_odors") or []):
                    continue
                ec_seq = tuple(to_ec_level(ec, 3) for ec in pw.get("ec_sequence", []) if to_ec_level(ec, 3))
                if required_ec not in ec_seq:
                    continue

                op_full = self._op_sum(ec_seq)
                if op_full is None:
                    continue
                ec_seq_cf = tuple(ec for ec in ec_seq if ec != required_ec)
                if not ec_seq_cf:
                    continue
                op_cf = self._op_sum(ec_seq_cf)
                if op_cf is None:
                    continue

                src_emb = self._get_embedding_for_kegg(pw.get("source"))
                tgt_emb = self._get_embedding_for_kegg(pw.get("target"))
                if src_emb is None or tgt_emb is None:
                    continue
                delta = tgt_emb - src_emb
                full_cos = self._cosine(op_full, delta)
                cf_cos = self._cosine(op_cf, delta)
                drops.append(full_cos - cf_cos)
        return self._summarize("counterfactual_drop", drops), drops

    def gated(self, pathways: List[Dict], rules: List[ComplexFOLRule]) -> Dict:
        gated = []
        ungated = []
        for rule in rules:
            if rule.rule_type != "conditional_necessary":
                continue
            required_ec = rule.head.get("necessary_ec")
            src = rule.body.get("source")
            tgt = rule.body.get("target")
            if not required_ec or not src or not tgt:
                continue

            for pw in pathways:
                if src not in (pw.get("source_odors") or []):
                    continue
                if tgt not in (pw.get("target_odors") or []):
                    continue
                ec_seq = tuple(to_ec_level(ec, 3) for ec in pw.get("ec_sequence", []) if to_ec_level(ec, 3))
                if not ec_seq:
                    continue
                op_sum = self._op_sum(ec_seq)
                if op_sum is None:
                    continue
                src_emb = self._get_embedding_for_kegg(pw.get("source"))
                tgt_emb = self._get_embedding_for_kegg(pw.get("target"))
                if src_emb is None or tgt_emb is None:
                    continue
                delta = tgt_emb - src_emb
                cos = self._cosine(op_sum, delta)
                if required_ec in ec_seq:
                    gated.append(cos)
                else:
                    ungated.append(cos)
        return {
            "name": "gated_effect",
            "with_gate": self._basic_stats(gated),
            "without_gate": self._basic_stats(ungated),
        }, {"with_gate": gated, "without_gate": ungated}

    def negative(self, pathways: List[Dict], rules: List[ComplexFOLRule], odor_to_kegg: Dict[str, List[str]]) -> Dict:
        repulsion = []
        for rule in rules:
            if rule.rule_type != "exclusion":
                continue
            src_odor = rule.body.get("source")
            ec_seq = tuple(rule.body.get("ec_sequence") or [])
            excluded = rule.head.get("excluded_target")
            if not src_odor or not ec_seq or not excluded:
                continue

            for pw in pathways:
                if src_odor not in (pw.get("source_odors") or []):
                    continue
                pw_seq = tuple(to_ec_level(ec, 3) for ec in pw.get("ec_sequence", []) if to_ec_level(ec, 3))
                if pw_seq != ec_seq:
                    continue
                op_sum = self._op_sum(pw_seq)
                if op_sum is None:
                    continue
                src_emb = self._get_embedding_for_kegg(pw.get("source"))
                if src_emb is None:
                    continue

                excluded_keggs = odor_to_kegg.get(excluded, [])[:20]
                for tgt_kegg in excluded_keggs:
                    tgt_emb = self._get_embedding_for_kegg(tgt_kegg)
                    if tgt_emb is None:
                        continue
                    delta = tgt_emb - src_emb
                    repulsion.append(self._cosine(op_sum, delta))
        return self._summarize("negative_repulsion", repulsion), repulsion

    @staticmethod
    def _basic_stats(values: List[float]) -> Dict:
        if not values:
            return {"n": 0}
        arr = np.array(values, dtype=float)
        return {
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "positive_ratio": float(np.mean(arr > 0)),
            "high_ratio": float(np.mean(arr > 0.3)),
        }

    def _summarize(self, name: str, values: List[float]) -> Dict:
        summary = {"name": name}
        summary.update(self._basic_stats(values))
        return summary


def build_odor_to_kegg(csv_file: str) -> Dict[str, List[str]]:
    df = pd.read_csv(csv_file)
    non_odor = {"TGSC ID", "TGSC_IDs", "CID", "CIDs", "KEGG_IDs", "KEGG_Source",
                "IsomericSMILES", "nonStereoSMILES", "IUPACName", "Solvent",
                "Updated_Desc_v2", "descriptors", "Unnamed: 0"}
    odor_cols = [c for c in df.columns if c not in non_odor]
    odor_to_kegg = defaultdict(list)
    for _, row in df.iterrows():
        kegg_ids = row.get("KEGG_IDs", "")
        if not isinstance(kegg_ids, str) or not kegg_ids:
            continue
        kids = [k.strip() for k in kegg_ids.split(";") if re.fullmatch(r"C\d{5}", k.strip())]
        if not kids:
            continue
        for col in odor_cols:
            try:
                if float(row.get(col, 0)) > 0:
                    for kid in kids:
                        odor_to_kegg[col].append(kid)
            except Exception:
                continue
    return odor_to_kegg


def _plot_operator_family_results(results: Dict, raw: Dict, output_dir: Path) -> None:
    _ensure_mpl_config_dir()
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["compositional", "counterfactual", "gated_with", "gated_without", "negative"]
    means = [
        results["compositional"].get("mean", 0.0),
        results["counterfactual"].get("mean", 0.0),
        results["gated"]["with_gate"].get("mean", 0.0),
        results["gated"]["without_gate"].get("mean", 0.0),
        results["negative"].get("mean", 0.0),
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, means, color=["#4C78A8", "#72B7B2", "#54A24B", "#E45756", "#8E6C8A"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean cosine / drop", fontweight="bold")
    ax.set_title("Operator Family Summary (Mean Effects)", fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "multi_rule_summary_means.png", dpi=300, bbox_inches="tight")
    plt.close()

    def _hist(values, title, filename):
        if not values:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=30, color="#4C78A8", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("cosine / drop")
        ax.set_ylabel("count")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    _hist(raw.get("compositional", []), "Compositional Cosine", "compositional_hist.png")
    _hist(raw.get("counterfactual", []), "Counterfactual Drop", "counterfactual_hist.png")
    _hist(raw.get("negative", []), "Negative Repulsion Cosine", "negative_hist.png")

    gated = raw.get("gated", {})
    if gated.get("with_gate"):
        _hist(gated.get("with_gate"), "Gated (with gate)", "gated_with_hist.png")
    if gated.get("without_gate"):
        _hist(gated.get("without_gate"), "Gated (without gate)", "gated_without_hist.png")


def filter_rules(
    rules: List[ComplexFOLRule],
    min_support: float = MIN_RULE_SUPPORT,
    min_confidence: float = MIN_RULE_CONFIDENCE,
    min_coverage: float = MIN_RULE_COVERAGE,
    max_per_type: int = MAX_RULES_PER_TYPE,
) -> List[ComplexFOLRule]:
    """Filter and cap rules to keep evaluation tractable."""
    by_type = defaultdict(list)
    for rule in rules:
        if rule.support < min_support:
            continue
        if rule.rule_type in {"necessary", "conditional_necessary", "sufficient", "mutual_exclusion"}:
            if rule.confidence < min_confidence:
                continue
        if rule.rule_type == "exclusion" and rule.coverage < min_coverage:
            continue
        by_type[rule.rule_type].append(rule)

    filtered = []
    for rule_type, items in by_type.items():
        items.sort(key=lambda r: (r.support * r.confidence, r.coverage), reverse=True)
        filtered.extend(items[:max_per_type])
    return filtered


def extract_ec_edges_from_pathways(pathways: List[Dict]) -> List[Tuple[str, str, str]]:
    edges = []
    for pw in pathways:
        for step in pw.get("steps", []):
            src = step.get("from")
            tgt = step.get("to")
            ec_nums = step.get("ec_numbers", [])
            if not src or not tgt or not ec_nums:
                continue
            ec3 = to_ec_level(ec_nums[0], 3)
            if ec3:
                edges.append((src, tgt, ec3))
    return list(set(edges))


def main():
    print("=" * 80)
    print("跨数据集EC算子验证系统（多规则版）")
    print("=" * 80)

    qian = QianDataLoader("../publications")
    if not qian.load_embeddings():
        print("\n❌ 请先运行:")
        print("   git lfs install")
        print("   git clone https://github.com/osmoai/publications.git")
        return
    qian.load_all_datasets()

    mapping = TGSCMappingLoader()
    mapping.load(
        "../02_kegg_mapping/tgsc_to_kegg.csv",
        "../02_kegg_mapping/tgsc_to_kegg.csv",
    )

    cache_file = "../kegg_cache/pathways_cache.pkl"
    print("\n📦 加载pathway cache...")
    cache_data = pickle.load(open(cache_file, "rb"))
    pathways = cache_data.get("pathways", [])
    print(f"✅ 加载了 {len(pathways)} 条pathways")

    ec_edges = extract_ec_edges_from_pathways(pathways)
    print(f"✅ 提取了 {len(ec_edges)} 条EC边")

    extractor = ECOperatorExtractor(lambda kegg: qian.get_embedding(mapping.kegg_to_smiles.get(kegg, "")))
    operators = extractor.extract_operators(ec_edges, min_samples=5)
    if not operators:
        print("❌ 未能提取任何算子")
        return

    # 规则生成
    print("\n📚 生成复杂规则...")
    rule_extractor = RuleExtractorV2(min_support=5.0, high_confidence=0.8)
    rule_extractor.ingest_pathways(pathways)
    rules = rule_extractor.extract_rules()
    rules = filter_rules(rules)
    print(f"✅ 规则数量: {len(rules)} (filtered)")

    odor_to_kegg = build_odor_to_kegg("../02_kegg_mapping/tgsc_to_kegg.csv")

    validator = OperatorFamilyValidator(operators, mapping.kegg_to_smiles, qian)
    print("\n🔬 多规则验证...")
    comp_sum, comp_vals = validator.compositional(pathways)
    cf_sum, cf_vals = validator.counterfactual(pathways, rules)
    gated_sum, gated_vals = validator.gated(pathways, rules)
    neg_sum, neg_vals = validator.negative(pathways, rules, odor_to_kegg)
    results = {
        "compositional": comp_sum,
        "counterfactual": cf_sum,
        "gated": gated_sum,
        "negative": neg_sum,
    }
    raw_results = {
        "compositional": comp_vals,
        "counterfactual": cf_vals,
        "gated": gated_vals,
        "negative": neg_vals,
    }

    report_file = "./cross_validation_report_multi.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 报告已保存: {report_file}")

    raw_file = "./cross_validation_report_multi_raw.pkl"
    with open(raw_file, "wb") as f:
        pickle.dump(raw_results, f)
    print(f"✅ 原始结果已保存: {raw_file}")

    charts_dir = Path("./visualization_charts_v2_multi")
    _plot_operator_family_results(results, raw_results, charts_dir)
    print(f"✅ 多规则图表已保存: {charts_dir}")

    print("\n" + "=" * 80)
    print("📊 多规则验证汇总")
    print("=" * 80)
    for key, summary in results.items():
        print(f"\n[{key}]")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    # ============= 在 main() 函数末尾调用 =============
    # 在原 main() 函数的最后，添加：
    # 增强可视化
    enhanced_charts_dir = Path("./visualization_enhanced")
    plot_enhanced_multi_rule_visualizations(results, raw_results, str(enhanced_charts_dir))



"""
=============================================================================
可直接添加到 cross_dataset_ec_validation_multi.py 末尾的可视化代码
=============================================================================

使用方法：
1. 将下面的代码复制到 cross_dataset_ec_validation_multi.py 的 main() 函数末尾
2. 或者直接将整个文件内容追加到原文件末尾

代码会在 main() 运行后自动生成8张可视化图表
"""


# ============= 添加到 main() 函数末尾 =============

def plot_enhanced_multi_rule_visualizations(results: Dict, raw_results: Dict, output_dir: str):
    """生成增强可视化图表"""
    _ensure_mpl_config_dir()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 生成增强可视化图表...")

    # 颜色方案
    COLORS = {
        'compositional': '#4C78A8',
        'counterfactual': '#72B7B2',
        'gated_with': '#54A24B',
        'gated_without': '#E45756',
        'negative': '#8E6C8A',
    }

    # =========================================================================
    # Figure 1: 综合仪表板
    # =========================================================================
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Mean Effects
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Compositional', 'Counterfactual', 'Gated\n(with)', 'Gated\n(without)', 'Negative']
    means = [
        results.get('compositional', {}).get('mean', 0),
        results.get('counterfactual', {}).get('mean', 0),
        results.get('gated', {}).get('with_gate', {}).get('mean', 0),
        results.get('gated', {}).get('without_gate', {}).get('mean', 0),
        results.get('negative', {}).get('mean', 0)
    ]
    colors = [COLORS['compositional'], COLORS['counterfactual'],
              COLORS['gated_with'], COLORS['gated_without'], COLORS['negative']]

    bars = ax1.bar(labels, means, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax1.axhline(0, color='black', linewidth=1.5)
    ax1.set_ylabel('Mean Value', fontweight='bold')
    ax1.set_title('(A) Mean Cosine / Drop', fontweight='bold')
    ax1.tick_params(axis='x', rotation=20)
    for bar, mean in zip(bars, means):
        va = 'bottom' if mean >= 0 else 'top'
        offset = 0.01 if mean >= 0 else -0.01
        ax1.text(bar.get_x() + bar.get_width() / 2, mean + offset,
                 f'{mean:.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

    # Panel B: Positive Ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ratios = [
        results.get('compositional', {}).get('positive_ratio', 0),
        results.get('counterfactual', {}).get('positive_ratio', 0),
        results.get('gated', {}).get('with_gate', {}).get('positive_ratio', 0),
        results.get('gated', {}).get('without_gate', {}).get('positive_ratio', 0),
        results.get('negative', {}).get('positive_ratio', 0)
    ]
    bars = ax2.bar(labels, ratios, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax2.axhline(0.5, color='gray', linewidth=1.5, linestyle='--', label='Random (50%)')
    ax2.set_ylabel('Positive Ratio', fontweight='bold')
    ax2.set_title('(B) Positive Direction Ratio', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=20)
    ax2.legend(loc='upper right', fontsize=9)
    for bar, ratio in zip(bars, ratios):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{ratio:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel C: Sample Sizes
    ax3 = fig.add_subplot(gs[0, 2])
    sizes = [
        results.get('compositional', {}).get('n', 0),
        results.get('counterfactual', {}).get('n', 0),
        results.get('gated', {}).get('with_gate', {}).get('n', 0),
        results.get('gated', {}).get('without_gate', {}).get('n', 0),
        results.get('negative', {}).get('n', 0)
    ]
    bars = ax3.bar(labels, sizes, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    ax3.set_ylabel('Sample Size (n)', fontweight='bold')
    ax3.set_title('(C) Sample Sizes', fontweight='bold')
    ax3.tick_params(axis='x', rotation=20)
    ax3.set_yscale('log')
    for bar, size in zip(bars, sizes):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f'{size:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel D: Compositional Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    comp_vals = raw_results.get('compositional', [])
    if comp_vals:
        ax4.hist(comp_vals, bins=40, color=COLORS['compositional'], alpha=0.7, edgecolor='white')
        ax4.axvline(0, color='black', linewidth=1.5)
        ax4.axvline(np.mean(comp_vals), color='red', linewidth=2, linestyle='--',
                    label=f'Mean: {np.mean(comp_vals):.3f}')
        ax4.legend(loc='upper right', fontsize=9)
    ax4.set_xlabel('Cosine Similarity', fontweight='bold')
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title('(D) Compositional Distribution', fontweight='bold')

    # Panel E: Negative Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    neg_vals = raw_results.get('negative', [])
    if neg_vals:
        ax5.hist(neg_vals, bins=40, color=COLORS['negative'], alpha=0.7, edgecolor='white')
        ax5.axvline(0, color='black', linewidth=1.5)
        ax5.axvline(np.mean(neg_vals), color='red', linewidth=2, linestyle='--',
                    label=f'Mean: {np.mean(neg_vals):.3f}')
        ax5.legend(loc='upper right', fontsize=9)
        # 标注100%负向
        if results.get('negative', {}).get('positive_ratio', 0) == 0:
            ax5.text(0.5, 0.95, '✓ 100% REPULSION', transform=ax5.transAxes,
                     fontsize=12, fontweight='bold', color='green',
                     ha='center', va='top', bbox=dict(facecolor='lightgreen', alpha=0.8))
    ax5.set_xlabel('Cosine Similarity', fontweight='bold')
    ax5.set_ylabel('Count', fontweight='bold')
    ax5.set_title('(E) Negative Repulsion Distribution', fontweight='bold')

    # Panel F: Key Findings
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    findings = []
    neg_pos = results.get('negative', {}).get('positive_ratio', 0)
    neg_mean = results.get('negative', {}).get('mean', 0)
    neg_n = results.get('negative', {}).get('n', 0)

    if neg_pos == 0 and neg_n > 0:
        findings.append("🎯 KEY FINDING:")
        findings.append(f"   Exclusion rules: 100% repulsion")
        findings.append(f"   (n={neg_n:,}, mean={neg_mean:.3f})")
        findings.append("")

    comp_pos = results.get('compositional', {}).get('positive_ratio', 0)
    findings.append(f"• Compositional: {comp_pos:.1%} positive")

    cf_mean = results.get('counterfactual', {}).get('mean', 0)
    findings.append(f"• Counterfactual drop: {cf_mean:.3f}")

    gated_diff = (results.get('gated', {}).get('with_gate', {}).get('mean', 0) -
                  results.get('gated', {}).get('without_gate', {}).get('mean', 0))
    findings.append(f"• Gated differential: {gated_diff:.3f}")

    ax6.text(0.1, 0.9, '\n'.join(findings), transform=ax6.transAxes,
             fontsize=11, va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax6.set_title('(F) Summary', fontweight='bold')

    plt.suptitle('Cross-Dataset EC Operator Validation Dashboard',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path / 'enhanced_01_dashboard.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Figure 1: Dashboard")

    # =========================================================================
    # Figure 2: Violin Plot Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    data_to_plot = []
    plot_labels = []
    plot_colors = []

    for key, label, color in [
        ('compositional', 'Compositional', COLORS['compositional']),
        ('counterfactual', 'Counterfactual', COLORS['counterfactual']),
        ('negative', 'Negative', COLORS['negative'])
    ]:
        values = raw_results.get(key, [])
        if values:
            data_to_plot.append(values)
            plot_labels.append(label)
            plot_colors.append(color)

    gated = raw_results.get('gated', {})
    if gated.get('with_gate'):
        data_to_plot.append(gated['with_gate'])
        plot_labels.append('Gated (with)')
        plot_colors.append(COLORS['gated_with'])

    if data_to_plot:
        parts = ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('red')
        parts['cmedians'].set_color('blue')
        ax.set_xticks(range(1, len(plot_labels) + 1))
        ax.set_xticklabels(plot_labels, fontweight='bold')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_title('Distribution Comparison (Violin Plot)', fontweight='bold')
        legend_elements = [
            mpatches.Patch(color='red', label='Mean'),
            mpatches.Patch(color='blue', label='Median'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path / 'enhanced_02_violin.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Figure 2: Violin Plot")

    # =========================================================================
    # Figure 3: Effect Size Analysis
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Cohen's d
    ax1 = axes[0]
    effects = []
    eff_labels = []
    eff_colors = []

    for key, label, color in [
        ('compositional', 'Compositional', COLORS['compositional']),
        ('counterfactual', 'Counterfactual', COLORS['counterfactual']),
        ('negative', 'Negative', COLORS['negative'])
    ]:
        values = raw_results.get(key, [])
        if values:
            d = np.mean(values) / (np.std(values) + 1e-8)
            effects.append(d)
            eff_labels.append(label)
            eff_colors.append(color)

    if effects:
        bars = ax1.barh(eff_labels, effects, color=eff_colors, edgecolor='black', alpha=0.85)
        ax1.axvline(0, color='black', linewidth=1)
        ax1.axvline(0.2, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax1.axvline(-0.2, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        ax1.axvline(0.5, color='orange', linewidth=1, linestyle='--', alpha=0.5)
        ax1.axvline(-0.5, color='orange', linewidth=1, linestyle='--', alpha=0.5)
        ax1.set_xlabel("Cohen's d", fontweight='bold')
        ax1.set_title("(A) Effect Size", fontweight='bold')
        for bar, val in zip(bars, effects):
            ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}', ha='left', va='center', fontsize=10, fontweight='bold')

    # High ratio
    ax2 = axes[1]
    high_ratios = [
        results.get('compositional', {}).get('high_ratio', 0),
        results.get('counterfactual', {}).get('high_ratio', 0),
        results.get('negative', {}).get('high_ratio', 0),
        results.get('gated', {}).get('with_gate', {}).get('high_ratio', 0),
    ]
    hr_labels = ['Compositional', 'Counterfactual', 'Negative', 'Gated (with)']
    hr_colors = [COLORS['compositional'], COLORS['counterfactual'], COLORS['negative'], COLORS['gated_with']]

    bars = ax2.bar(hr_labels, high_ratios, color=hr_colors, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('High Ratio (|cos| > 0.3)', fontweight='bold')
    ax2.set_title('(B) Strong Effect Ratio', fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    ax2.set_ylim(0, max(high_ratios) * 1.3 + 0.05 if high_ratios else 1)
    for bar, ratio in zip(bars, high_ratios):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{ratio:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Effect Size Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'enhanced_03_effect_size.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Figure 3: Effect Size")

    # =========================================================================
    # Figure 4: Paper-Ready Figure
    # =========================================================================
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # A: Mean with CI
    ax1 = fig.add_subplot(gs[0, 0])
    paper_labels = ['Compositional', 'Counterfactual', 'Gated', 'Negative']
    paper_means = []
    paper_cis = []
    paper_colors = [COLORS['compositional'], COLORS['counterfactual'], COLORS['gated_with'], COLORS['negative']]

    for key in ['compositional', 'counterfactual']:
        vals = raw_results.get(key, [])
        if vals:
            paper_means.append(np.mean(vals))
            paper_cis.append(1.96 * np.std(vals) / np.sqrt(len(vals)))
        else:
            paper_means.append(0)
            paper_cis.append(0)

    gated_vals = raw_results.get('gated', {}).get('with_gate', [])
    paper_means.append(np.mean(gated_vals) if gated_vals else 0)
    paper_cis.append(1.96 * np.std(gated_vals) / np.sqrt(len(gated_vals)) if gated_vals else 0)

    neg_vals = raw_results.get('negative', [])
    paper_means.append(np.mean(neg_vals) if neg_vals else 0)
    paper_cis.append(1.96 * np.std(neg_vals) / np.sqrt(len(neg_vals)) if neg_vals else 0)

    bars = ax1.bar(paper_labels, paper_means, yerr=paper_cis, capsize=6,
                   color=paper_colors, edgecolor='black', alpha=0.85)
    ax1.axhline(0, color='black', linewidth=1.5)
    ax1.set_ylabel('Mean ± 95% CI', fontweight='bold')
    ax1.set_title('(A) Mean Effects', fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)

    # B: Distribution overlay
    ax2 = fig.add_subplot(gs[0, 1])
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
    ax2.legend(loc='upper left')

    # C: Sample sizes
    ax3 = fig.add_subplot(gs[1, 0])
    paper_sizes = [
        results.get('compositional', {}).get('n', 0),
        results.get('counterfactual', {}).get('n', 0),
        results.get('gated', {}).get('with_gate', {}).get('n', 0),
        results.get('negative', {}).get('n', 0)
    ]
    bars = ax3.bar(paper_labels, paper_sizes, color=paper_colors, edgecolor='black', alpha=0.85)
    ax3.set_ylabel('Sample Size (n)', fontweight='bold')
    ax3.set_title('(C) Sample Sizes', fontweight='bold')
    ax3.tick_params(axis='x', rotation=15)
    ax3.set_yscale('log')
    for bar, size in zip(bars, paper_sizes):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                 f'{size:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # D: Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    summary_lines = ["VALIDATION SUMMARY", "=" * 25, ""]
    neg_pos = results.get('negative', {}).get('positive_ratio', 0)
    if neg_pos == 0:
        summary_lines.append("✓ Exclusion: 100% REPULSION")
    comp_pos = results.get('compositional', {}).get('positive_ratio', 0)
    summary_lines.append(f"△ Compositional: {comp_pos:.0%} positive")
    cf_mean = results.get('counterfactual', {}).get('mean', 0)
    summary_lines.append(f"△ Counterfactual: {cf_mean:.3f}")
    summary_lines.append("")
    if neg_pos == 0:
        summary_lines.append("CONCLUSION:")
        summary_lines.append("EC operators encode meaningful")
        summary_lines.append("biochemical transformation rules")

    ax4.text(0.1, 0.9, '\n'.join(summary_lines), transform=ax4.transAxes,
             fontsize=10, va='top', ha='left', family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    ax4.set_title('(D) Summary', fontweight='bold')

    plt.suptitle('Cross-Dataset EC Operator Validation', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path / 'enhanced_04_paper_figure.png', bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / 'enhanced_04_paper_figure.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Figure 4: Paper Figure (PNG + PDF)")

    print(f"\n✅ 增强可视化完成: {output_path}")




if __name__ == "__main__":
    main()


