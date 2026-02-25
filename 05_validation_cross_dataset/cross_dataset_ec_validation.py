#!/usr/bin/env python3
"""
跨数据集EC算子验证系统
======================

整合你的FOL规则提取系统 + Qian et al. 2023的跨物种数据集

目标：验证从TGSC提取的EC算子在跨物种数据集上的泛化能力

作者：整合自用户代码 + Qian数据
"""

import pandas as pd
import numpy as np
import pickle
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

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


# =============================================================================
# Part 1: 数据加载器
# =============================================================================

class QianDataLoader:
    """加载Qian et al. 2023的数据"""
    
    def __init__(self, publications_dir: str = "../publications"):
        """
        Args:
            publications_dir: clone的osmoai/publications仓库路径
        """
        self.base_dir = Path(publications_dir)
        self.data_dir = self.base_dir / "qian_et_al_2023" / "predictive_performance" / "data"
        
        self.embeddings = None
        self.embeddings_dict = {}  # smiles -> 256-dim vector
        self.datasets = {}  # dataset_name -> DataFrame
        
    def load_embeddings(self) -> bool:
        """
        加载embeddings.csv
        
        格式: smiles (index), dim0, dim1, ..., dim255
        """
        emb_file = self.data_dir / "embeddings.csv"
        
        if not emb_file.exists():
            print(f"❌ embeddings.csv不存在: {emb_file}")
            print("   请先运行: git clone https://github.com/osmoai/publications.git")
            return False
        
        print(f"📦 加载embeddings.csv...")
        self.embeddings = pd.read_csv(emb_file, index_col=0)
        
        print(f"   Shape: {self.embeddings.shape}")
        print(f"   Columns: dim0 to dim{self.embeddings.shape[1]-1}")
        
        # 构建字典: smiles -> embedding
        for smiles in self.embeddings.index:
            emb = self.embeddings.loc[smiles].values.astype(np.float32)
            self.embeddings_dict[smiles] = emb
            
            # 如果有RDKit，补充isomeric和non-isomeric版本
            if HAS_RDKIT:
                can_iso = self._canonicalize(smiles, isomeric=True)
                if can_iso and can_iso != smiles:
                    self.embeddings_dict[can_iso] = emb
                can_non = self._canonicalize_non_isomeric(smiles)
                if can_non and can_non != smiles:
                    self.embeddings_dict[can_non] = emb
        
        print(f"✅ 加载了 {len(self.embeddings)} 个分子的embeddings")
        return True
    
    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        加载特定数据集
        
        Args:
            dataset_name: 如 'a_MacWilliam_et_al', 'h_Keller_et_al' 等
        """
        data_file = self.data_dir / dataset_name / "data.csv"
        
        if not data_file.exists():
            print(f"❌ 数据集不存在: {data_file}")
            return None
        
        df = pd.read_csv(data_file)
        self.datasets[dataset_name] = df
        
        print(f"✅ 加载数据集 {dataset_name}: {len(df)} 样本")
        print(f"   列: {df.columns.tolist()}")
        
        return df

    def _discover_dataset_names(self) -> List[str]:
        """
        自动发现数据集：
        1) data/<dataset>/data.csv
        2) data/<group>/<subdataset>/data.csv
        """
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

            # 支持一层嵌套目录（例如 e_Missbach_et_al/*/data.csv）
            for child in sorted(item.iterdir()):
                if child.is_dir() and (child / "data.csv").exists():
                    names.append(f"{item.name}/{child.name}")

        return names
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """加载所有可用数据集"""
        dataset_dirs = self._discover_dataset_names()
        
        print(f"\n📦 加载所有数据集...")
        print(f"   自动发现 {len(dataset_dirs)} 个数据集路径")
        
        for name in dataset_dirs:
            try:
                self.load_dataset(name)
            except Exception as e:
                print(f"   ⚠️ {name}: {e}")
        
        print(f"\n✅ 成功加载 {len(self.datasets)} 个数据集")
        return self.datasets
    
    def get_embedding(self, smiles: str) -> Optional[np.ndarray]:
        """获取分子的embedding"""
        # 直接查找
        if smiles in self.embeddings_dict:
            return self.embeddings_dict[smiles]
        
        # 尝试canonical形式
        if HAS_RDKIT:
            can_iso = self._canonicalize(smiles, isomeric=True)
            if can_iso and can_iso in self.embeddings_dict:
                return self.embeddings_dict[can_iso]
            can_non = self._canonicalize_non_isomeric(smiles)
            if can_non and can_non in self.embeddings_dict:
                return self.embeddings_dict[can_non]
        
        return None
    
    @staticmethod
    def _canonicalize(smiles: str, isomeric: bool = True) -> Optional[str]:
        """标准化SMILES（可选保留/移除立体信息）"""
        if not HAS_RDKIT:
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
        except Exception:
            pass
        return None

    def _canonicalize_non_isomeric(self, smiles: str) -> Optional[str]:
        """统一到非isomeric SMILES（与OpenPOM一致）"""
        return normalize_non_isomeric_smiles(smiles)


class TGSCDataLoader:
    """加载你的TGSC数据和pathway cache"""
    
    def __init__(self, cache_dir: str = "../kegg_cache"):
        self.cache_dir = Path(cache_dir)
        self.pathways = []
        self.ec_operators = {}  # ec_code -> delta_vector (256-dim)
        self.tgsc_embeddings = {}  # smiles -> embedding
        self.kegg_to_smiles = {}
        self.mapping_loaded = False
        
    def load_pathway_cache(self) -> List[Dict]:
        """加载pathway cache"""
        cache_file = self.cache_dir / "pathways_cache.pkl"
        
        if not cache_file.exists():
            print(f"❌ Pathway cache不存在: {cache_file}")
            print("   请先运行你的main()函数生成pathways")
            return []
        
        print(f"📦 加载pathway cache...")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.pathways = cache_data.get('pathways', [])
        metadata = cache_data.get('metadata', {})
        
        print(f"✅ 加载了 {len(self.pathways)} 条pathways")
        print(f"   参数: {cache_data.get('parameters', {})}")
        
        return self.pathways
    
    def load_tgsc_kegg_mapping(
        self,
        csv_file: str = "../02_kegg_mapping/tgsc_to_kegg.csv",
        merged_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """加载TGSC到KEGG的映射，并联动merged_dataset_with_kegg.csv的nonStereoSMILES"""
        df = pd.read_csv(csv_file)
        print(f"✅ 加载TGSC映射: {len(df)} 化合物")

        merged_added = 0
        merged_missing = 0
        if merged_csv and Path(merged_csv).exists():
            merged_df = pd.read_csv(merged_csv)
            if "KEGG_IDs" in merged_df.columns and "nonStereoSMILES" in merged_df.columns:
                for _, row in merged_df.iterrows():
                    raw_smiles = str(row["nonStereoSMILES"]).strip()
                    if not raw_smiles or raw_smiles == "nan":
                        merged_missing += 1
                        continue
                    normalized = self._to_non_isomeric_smiles(raw_smiles)
                    if not normalized:
                        merged_missing += 1
                        continue
                    kegg_ids = self._split_kegg_ids(str(row["KEGG_IDs"]))
                    for kid in kegg_ids:
                        if kid and kid not in self.kegg_to_smiles:
                            self.kegg_to_smiles[kid] = normalized
                            merged_added += 1
                print(f"✅ merged_dataset_with_kegg映射: +{merged_added} 个KEGG ID")
                if merged_missing > 0:
                    print(f"⚠️ merged_dataset_with_kegg 无法解析SMILES行数: {merged_missing}")
            else:
                print("⚠️ merged_dataset_with_kegg.csv缺少KEGG_IDs或nonStereoSMILES列")
        elif merged_csv:
            print(f"⚠️ merged_dataset_with_kegg.csv不存在: {merged_csv}")

        smiles_col = None
        for candidate in ["nonStereoSMILES", "IsomericSMILES", "SMILES", "smiles"]:
            if candidate in df.columns:
                smiles_col = candidate
                break

        if not smiles_col:
            print("❌ 未找到SMILES列（nonStereoSMILES/IsomericSMILES/SMILES/smiles）")
            return df

        if "KEGG_IDs" not in df.columns:
            print("❌ 未找到KEGG_IDs列")
            return df

        missing_smiles = 0
        tgsc_added = 0
        for _, row in df.iterrows():
            raw_smiles = str(row[smiles_col]).strip()
            if not raw_smiles or raw_smiles == "nan":
                missing_smiles += 1
                continue

            normalized = self._to_non_isomeric_smiles(raw_smiles)
            if not normalized:
                missing_smiles += 1
                continue

            kegg_ids = self._split_kegg_ids(str(row["KEGG_IDs"]))
            for kid in kegg_ids:
                if kid and kid not in self.kegg_to_smiles:
                    self.kegg_to_smiles[kid] = normalized
                    tgsc_added += 1

        self.mapping_loaded = True
        print(f"✅ KEGG->SMILES映射: {len(self.kegg_to_smiles)} 个KEGG ID")
        if tgsc_added > 0:
            print(f"   其中TGSC新增: {tgsc_added}")
        if missing_smiles > 0:
            print(f"⚠️ 无法解析SMILES的行数: {missing_smiles}")
        return df
    
    def extract_ec_edges(self) -> List[Tuple[str, str, str]]:
        """
        从pathways提取EC边
        
        Returns:
            List of (source_compound, target_compound, ec_code)
        """
        edges = []
        
        missing_mapping = 0
        used_mapping = 0

        for pw in self.pathways:
            for step in pw.get('steps', []):
                src = step.get('from')
                tgt = step.get('to')
                ec_nums = step.get('ec_numbers', [])
                
                if src and tgt and ec_nums:
                    src_smiles = self._resolve_compound_id(src)
                    tgt_smiles = self._resolve_compound_id(tgt)
                    if not src_smiles or not tgt_smiles:
                        missing_mapping += 1
                        continue
                    used_mapping += 1
                    # 取第一个EC（三级）
                    ec = ec_nums[0]
                    ec3 = self._to_ec_level(ec, 3)
                    if ec3:
                        edges.append((src_smiles, tgt_smiles, ec3))
        
        # 去重
        edges = list(set(edges))
        print(f"✅ 提取了 {len(edges)} 条EC边")
        if self.mapping_loaded:
            print(f"   ✅ 成功映射: {used_mapping}，未映射跳过: {missing_mapping}")
        
        return edges

    def _resolve_compound_id(self, value: str) -> Optional[str]:
        """将KEGG ID解析为SMILES；若不是KEGG ID则返回规范化SMILES或原值"""
        if not value:
            return None
        value = value.strip()

        if self._is_kegg_id(value):
            return self.kegg_to_smiles.get(value)

        return self._to_non_isomeric_smiles(value)

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
            if TGSCDataLoader._is_kegg_id(part):
                cleaned.append(part)
        return cleaned

    @staticmethod
    def _is_kegg_id(value: str) -> bool:
        return bool(re.fullmatch(r"C\d{5}", value))

    @staticmethod
    def _to_non_isomeric_smiles(smiles: str) -> Optional[str]:
        return normalize_non_isomeric_smiles(smiles)
    
    @staticmethod
    def _to_ec_level(ec: str, level: int = 3) -> str:
        """截取EC到指定级别"""
        if not ec or '.' not in ec:
            return ''
        parts = ec.split('.')
        if len(parts) < level:
            return ''
        result = parts[:level]
        if result[-1] == '-':
            return ''
        return '.'.join(result)


# =============================================================================
# Part 2: EC算子提取器
# =============================================================================

class ECOperatorExtractor:
    """
    从pathway数据中提取EC算子
    
    EC算子定义: Op_EC = mean(target_emb - source_emb) for all (src, tgt) via EC
    """
    
    def __init__(self, embedding_getter):
        """
        Args:
            embedding_getter: callable, smiles -> 256-dim embedding or None
        """
        self.get_embedding = embedding_getter
        self.operators = {}  # ec_code -> np.array (256,)
        self.operator_stats = {}  # ec_code -> {n_samples, std, etc.}
        
    def extract_operators(
        self,
        ec_edges: List[Tuple[str, str, str]],
        min_samples: int = 5,
        min_mean_op_norm: float = 0.05,
        min_mean_delta_norm: float = 0.05,
    ) -> Dict[str, np.ndarray]:
        """
        提取EC算子
        
        Args:
            ec_edges: [(src_compound, tgt_compound, ec_code), ...]
            min_samples: 最少样本数
            min_mean_op_norm: 算子平均向量最小范数
            min_mean_delta_norm: 平均delta最小范数
            
        Returns:
            {ec_code: 256-dim operator vector}
        """
        print(f"\n📐 提取EC算子...")
        
        # 按EC分组收集delta向量
        ec_deltas = defaultdict(list)
        
        for src, tgt, ec in tqdm(ec_edges, desc="计算delta向量"):
            src_emb = self.get_embedding(src)
            tgt_emb = self.get_embedding(tgt)
            
            if src_emb is not None and tgt_emb is not None:
                delta = tgt_emb - src_emb
                ec_deltas[ec].append(delta)
        
        # 计算平均算子
        for ec, deltas in ec_deltas.items():
            if len(deltas) >= min_samples:
                deltas_arr = np.array(deltas)
                mean_op = np.mean(deltas_arr, axis=0)
                std_op = np.std(deltas_arr, axis=0)
                mean_delta_norm = float(np.mean(np.linalg.norm(deltas_arr, axis=1)))
                mean_op_norm = float(np.linalg.norm(mean_op))

                if mean_op_norm < min_mean_op_norm or mean_delta_norm < min_mean_delta_norm:
                    continue
                
                self.operators[ec] = mean_op
                self.operator_stats[ec] = {
                    'n_samples': len(deltas),
                    'mean_magnitude': mean_op_norm,
                    'std_magnitude': np.mean(np.linalg.norm(std_op)),
                    'consistency': 1.0 - np.mean(std_op) / (np.mean(np.abs(mean_op)) + 1e-8)
                }
        
        print(f"✅ 提取了 {len(self.operators)} 个EC算子")
        
        # 显示Top算子
        top_ops = sorted(self.operator_stats.items(), 
                        key=lambda x: x[1]['n_samples'], reverse=True)[:10]
        print(f"\n   Top 10 算子 (按样本数):")
        for ec, stats in top_ops:
            print(f"     {ec}: n={stats['n_samples']}, "
                  f"mag={stats['mean_magnitude']:.3f}, "
                  f"consistency={stats['consistency']:.3f}")
        
        return self.operators
    
    def get_operator(self, ec_code: str) -> Optional[np.ndarray]:
        """获取EC算子"""
        return self.operators.get(ec_code)


# =============================================================================
# Part 3: 跨数据集验证器
# =============================================================================

@dataclass
class ValidationResult:
    """验证结果"""
    dataset_name: str
    n_molecules: int
    n_molecules_with_embedding: int
    n_ec_edges_testable: int
    metrics: Dict
    details: List[Dict]


class CrossDatasetValidator:
    """跨数据集验证器"""
    
    def __init__(self, qian_loader: QianDataLoader, 
                 ec_operators: Dict[str, np.ndarray]):
        self.qian = qian_loader
        self.operators = ec_operators
        self.results = {}
        
    def validate_on_dataset(self, dataset_name: str, 
                           ec_edges: List[Tuple[str, str, str]]) -> ValidationResult:
        """
        在特定数据集上验证EC算子
        
        验证方法:
        1. 找数据集中有embedding的分子
        2. 找这些分子参与的EC边
        3. 计算预测位置 vs 实际位置的一致性
        """
        print(f"\n🔬 验证数据集: {dataset_name}")
        
        # 获取数据集
        if dataset_name not in self.qian.datasets:
            df = self.qian.load_dataset(dataset_name)
            if df is None:
                return None
        
        df = self.qian.datasets[dataset_name]
        
        # 找SMILES列
        smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
        if smiles_col not in df.columns:
            print(f"   ❌ 未找到SMILES列")
            return None
        
        # 获取数据集中所有分子的embedding
        dataset_molecules = set(df[smiles_col].dropna().unique())
        molecules_with_emb = {}
        
        for smi in dataset_molecules:
            emb = self.qian.get_embedding(smi)
            if emb is not None:
                molecules_with_emb[smi] = emb
        
        print(f"   分子数: {len(dataset_molecules)}")
        print(f"   有embedding: {len(molecules_with_emb)}")
        
        # 筛选可测试的EC边
        testable_edges = []
        for src, tgt, ec in ec_edges:
            # 检查src和tgt是否都在当前数据集中且有embedding
            src_in = src in molecules_with_emb
            tgt_in = tgt in molecules_with_emb
            has_op = ec in self.operators
            
            if src_in and tgt_in and has_op:
                testable_edges.append((src, tgt, ec))
        
        print(f"   可测试EC边: {len(testable_edges)}")
        
        if not testable_edges:
            return ValidationResult(
                dataset_name=dataset_name,
                n_molecules=len(dataset_molecules),
                n_molecules_with_embedding=len(molecules_with_emb),
                n_ec_edges_testable=0,
                metrics={'note': 'No testable edges'},
                details=[]
            )
        
        # 计算验证指标
        cosine_sims = []
        magnitudes = []
        details = []
        
        for src, tgt, ec in testable_edges:
            src_emb = molecules_with_emb[src]
            tgt_emb = molecules_with_emb[tgt]
            op = self.operators[ec]
            
            # 预测位置
            predicted = src_emb + op
            
            # 实际delta
            actual_delta = tgt_emb - src_emb
            
            # 计算cosine similarity
            cos_sim = np.dot(op, actual_delta) / (
                np.linalg.norm(op) * np.linalg.norm(actual_delta) + 1e-8
            )
            
            # 计算magnitude ratio
            pred_mag = np.linalg.norm(op)
            actual_mag = np.linalg.norm(actual_delta)
            mag_ratio = min(pred_mag, actual_mag) / (max(pred_mag, actual_mag) + 1e-8)
            
            cosine_sims.append(cos_sim)
            magnitudes.append(mag_ratio)
            
            details.append({
                'src': src, 'tgt': tgt, 'ec': ec,
                'cosine_sim': float(cos_sim),
                'magnitude_ratio': float(mag_ratio)
            })
        
        # 汇总指标
        metrics = {
            'mean_cosine_sim': float(np.mean(cosine_sims)),
            'std_cosine_sim': float(np.std(cosine_sims)),
            'mean_magnitude_ratio': float(np.mean(magnitudes)),
            'positive_cosine_ratio': float(np.mean(np.array(cosine_sims) > 0)),
            'high_cosine_ratio': float(np.mean(np.array(cosine_sims) > 0.3))
        }
        
        print(f"   📊 结果:")
        print(f"      Mean Cosine Sim: {metrics['mean_cosine_sim']:.4f}")
        print(f"      Positive Ratio: {metrics['positive_cosine_ratio']:.2%}")
        print(f"      High (>0.3) Ratio: {metrics['high_cosine_ratio']:.2%}")
        
        result = ValidationResult(
            dataset_name=dataset_name,
            n_molecules=len(dataset_molecules),
            n_molecules_with_embedding=len(molecules_with_emb),
            n_ec_edges_testable=len(testable_edges),
            metrics=metrics,
            details=details
        )
        
        self.results[dataset_name] = result
        return result
    
    def validate_all_datasets(self, ec_edges: List[Tuple[str, str, str]]) -> Dict:
        """在所有数据集上验证"""
        print("\n" + "=" * 80)
        print("跨数据集验证")
        print("=" * 80)
        
        all_results = {}
        
        for dataset_name in self.qian.datasets.keys():
            result = self.validate_on_dataset(dataset_name, ec_edges)
            if result:
                all_results[dataset_name] = result
        
        return all_results
    
    def generate_report(self, output_file: str = "cross_validation_report.json"):
        """生成验证报告"""
        report = {
            'summary': {
                'total_datasets': len(self.results),
                'datasets_with_testable_edges': sum(
                    1 for r in self.results.values() if r.n_ec_edges_testable > 0
                )
            },
            'datasets': {}
        }
        
        for name, result in self.results.items():
            report['datasets'][name] = {
                'n_molecules': result.n_molecules,
                'n_with_embedding': result.n_molecules_with_embedding,
                'n_testable_edges': result.n_ec_edges_testable,
                'metrics': result.metrics
            }
        
        # 计算整体指标
        all_cosines = []
        for result in self.results.values():
            if result.details:
                all_cosines.extend([d['cosine_sim'] for d in result.details])
        
        if all_cosines:
            report['summary']['overall_mean_cosine'] = float(np.mean(all_cosines))
            report['summary']['overall_positive_ratio'] = float(np.mean(np.array(all_cosines) > 0))
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ 报告已保存: {output_file}")
        return report


# =============================================================================
# Part 4: 行为预测实验
# =============================================================================

class BehaviorPredictionExperiment:
    """
    实验：EC算子能否预测行为变化？
    
    假设：如果EC变换编码了生物学意义，那么：
    - 应用算子后的位置应该与目标行为相关
    - 不同EC应该导向不同的行为类别
    """
    
    def __init__(self, qian_loader: QianDataLoader,
                 ec_operators: Dict[str, np.ndarray]):
        self.qian = qian_loader
        self.operators = ec_operators
        
    def analyze_operator_behavior_correlation(self, dataset_name: str):
        """
        分析算子方向与行为标签的相关性
        """
        print(f"\n🧪 行为预测实验: {dataset_name}")
        
        df = self.qian.datasets.get(dataset_name)
        if df is None:
            print("   ❌ 数据集未加载")
            return None
        
        # 获取SMILES和标签列
        smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
        label_cols = [c for c in df.columns if c != smiles_col]
        
        if not label_cols:
            print("   ❌ 未找到标签列")
            return None
        
        label_col = label_cols[0]  # 取第一个标签列
        print(f"   标签列: {label_col}")
        
        # 构建embedding矩阵和标签向量
        emb_list = []
        labels = []
        smiles_list = []
        
        for _, row in df.iterrows():
            smi = row[smiles_col]
            label = row[label_col]
            
            emb = self.qian.get_embedding(smi)
            if emb is not None and pd.notna(label):
                emb_list.append(emb)
                labels.append(label)
                smiles_list.append(smi)
        
        if not emb_list:
            print("   ❌ 没有有效的embedding-label对")
            return None
        
        emb_matrix = np.array(emb_list)
        labels = np.array(labels)
        
        print(f"   有效样本: {len(labels)}")
        print(f"   标签分布: {Counter(labels)}")
        
        # 对每个算子，分析它指向的方向与标签的关系
        results = {}
        
        for ec, op in self.operators.items():
            # 应用算子
            shifted = emb_matrix + op
            
            # 计算移动方向与原始位置的差异
            # 简单方法：看shifted位置最近邻的标签分布
            from sklearn.neighbors import NearestNeighbors
            
            nn = NearestNeighbors(n_neighbors=min(5, len(emb_matrix)))
            nn.fit(emb_matrix)
            
            distances, indices = nn.kneighbors(shifted)
            neighbor_labels = labels[indices]
            
            # 计算标签变化趋势
            mean_neighbor_label = np.mean(neighbor_labels, axis=1)
            original_vs_shifted = np.mean(mean_neighbor_label) - np.mean(labels)
            
            results[ec] = {
                'mean_shift': float(original_vs_shifted),
                'neighbor_label_mean': float(np.mean(mean_neighbor_label)),
                'original_label_mean': float(np.mean(labels))
            }
        
        # 排序显示
        sorted_results = sorted(results.items(), 
                               key=lambda x: abs(x[1]['mean_shift']), 
                               reverse=True)
        
        print(f"\n   Top 10 影响行为的EC算子:")
        for ec, stats in sorted_results[:10]:
            direction = "↑" if stats['mean_shift'] > 0 else "↓"
            print(f"     {ec}: shift={stats['mean_shift']:+.4f} {direction}")
        
        return results


# =============================================================================
# Part 5: 主程序
# =============================================================================

def main():
    """
    主程序：跨数据集EC算子验证
    
    前提条件：
    1. 已clone osmoai/publications到 ./publications
    2. 已运行你的main()生成 ./kegg_cache/pathways_cache.pkl
    """
    print("=" * 80)
    print("跨数据集EC算子验证系统")
    print("=" * 80)
    
    # =========================================================================
    # Step 1: 加载Qian数据
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: 加载Qian et al.数据")
    print("=" * 80)
    
    qian = QianDataLoader("../publications")
    
    if not qian.load_embeddings():
        print("\n❌ 请先运行:")
        print("   git lfs install")
        print("   git clone https://github.com/osmoai/publications.git")
        return
    
    qian.load_all_datasets()
    
    # =========================================================================
    # Step 2: 加载你的pathway数据
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: 加载TGSC Pathway数据")
    print("=" * 80)
    
    tgsc = TGSCDataLoader("../kegg_cache")
    pathways = tgsc.load_pathway_cache()
    
    if not pathways:
        print("\n❌ 请先运行你的main()函数生成pathways")
        return

    try:
        tgsc.load_tgsc_kegg_mapping(
            "../02_kegg_mapping/tgsc_to_kegg.csv",
        )
    except Exception as exc:
        print(f"⚠️ 无法加载tgsc_to_kegg.csv: {exc}")

    ec_edges = tgsc.extract_ec_edges()
    
    # =========================================================================
    # Step 3: 提取EC算子
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: 提取EC算子")
    print("=" * 80)
    
    # 使用Qian的embedding作为统一空间
    extractor = ECOperatorExtractor(qian.get_embedding)
    operators = extractor.extract_operators(ec_edges, min_samples=5)
    
    if not operators:
        print("❌ 未能提取任何算子")
        return
    
    # =========================================================================
    # Step 4: 跨数据集验证
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: 跨数据集验证")
    print("=" * 80)
    
    validator = CrossDatasetValidator(qian, operators)
    results = validator.validate_all_datasets(ec_edges)
    
    # =========================================================================
    # Step 5: 行为预测实验
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: 行为预测实验")
    print("=" * 80)
    
    behavior_exp = BehaviorPredictionExperiment(qian, operators)
    
    # 在MacWilliam数据集上测试（Drosophila行为）
    if 'a_MacWilliam_et_al' in qian.datasets:
        behavior_results = behavior_exp.analyze_operator_behavior_correlation(
            'a_MacWilliam_et_al'
        )
    
    # =========================================================================
    # Step 6: 生成报告
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: 生成报告")
    print("=" * 80)
    
    report = validator.generate_report("cross_validation_report.json")
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("📊 验证汇总")
    print("=" * 80)
    
    print(f"\n数据集数量: {len(results)}")
    
    if 'overall_mean_cosine' in report.get('summary', {}):
        print(f"整体平均Cosine: {report['summary']['overall_mean_cosine']:.4f}")
        print(f"正向比例: {report['summary']['overall_positive_ratio']:.2%}")
    
    print("\n各数据集结果:")
    for name, result in sorted(results.items(), 
                               key=lambda x: x[1].metrics.get('mean_cosine_sim', 0),
                               reverse=True):
        if result.n_ec_edges_testable > 0:
            print(f"  {name}:")
            print(f"    可测试边: {result.n_ec_edges_testable}")
            print(f"    Mean Cosine: {result.metrics['mean_cosine_sim']:.4f}")
            print(f"    Positive Ratio: {result.metrics['positive_cosine_ratio']:.2%}")
    
    print("\n" + "=" * 80)
    print("🎉 验证完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
