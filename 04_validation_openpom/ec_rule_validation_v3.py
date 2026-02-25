#!/usr/bin/env python3
"""
EC Rule Validation in OpenPOM Embedding Space (V2)
===================================================

验证 EC 规则在 OpenPOM 气味嵌入空间中的几何指纹。

核心设计：
- 边 = 端到端路径 (气味化合物A → 气味化合物B)
- Δ = f(B) - f(A) 在嵌入空间的位移
- 多种分组策略验证一致性

分组策略：
1. ec_sequence: 按EC序列分组（层级可调 1/2/3）
2. odor_pair: 按(源气味集合, 目标气味集合)分组
3. source_odor_first_ec: 按(源气味, 第一个EC)分组
4. target_odor_last_ec: 按(目标气味, 最后一个EC)分组
5. full_combination: 按(源气味, first_ec, 目标气味, last_ec)分组

Author: For EC-OpenPOM cross-validation
"""

import os
import json
import pickle
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, FrozenSet, Any
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings('ignore')
np.random.seed(42)

# Plotting configuration
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)


# ============================================================================
# PART 1: Data Structures
# ============================================================================

@dataclass
class EndToEndEdge:
    """
    端到端路径边：从气味化合物A到气味化合物B的完整路径
    """
    # 化合物信息
    source_compound: str  # KEGG ID (e.g., C00001)
    target_compound: str  # KEGG ID
    source_smiles: Optional[str] = None
    target_smiles: Optional[str] = None

    # 气味信息（多标签）
    source_odors: Tuple[str, ...] = ()  # e.g., ('mint', 'green', 'fresh')
    target_odors: Tuple[str, ...] = ()  # e.g., ('woody', 'herbal')

    # EC序列信息
    ec_sequence: Tuple[str, ...] = ()  # e.g., ('1.1.1', '2.3.4', '4.2.1')
    path_length: int = 0

    # 路径元数据
    pathway_id: int = 0
    odor_similarity: float = 0.0

    def get_ec_at_level(self, level: int = 3) -> Tuple[str, ...]:
        """获取指定层级的EC序列"""
        result = []
        for ec in self.ec_sequence:
            truncated = truncate_ec(ec, level)
            if truncated:
                result.append(truncated)
        return tuple(result)

    @property
    def first_ec(self) -> str:
        return self.ec_sequence[0] if self.ec_sequence else 'unknown'

    @property
    def last_ec(self) -> str:
        return self.ec_sequence[-1] if self.ec_sequence else 'unknown'

    @property
    def source_odor_set(self) -> FrozenSet[str]:
        return frozenset(self.source_odors)

    @property
    def target_odor_set(self) -> FrozenSet[str]:
        return frozenset(self.target_odors)


@dataclass
class GroupStatistics:
    """一个分组的统计信息"""
    group_key: Any
    n_edges: int
    delta_vectors: np.ndarray  # (n_edges, embedding_dim)
    mean_delta: np.ndarray  # (embedding_dim,)
    delta_dispersion: float  # 组内方差
    direction_consistency: float  # 方向一致性 (pairwise cos sim)
    mean_magnitude: float  # 平均位移幅度

    # 可选的详细信息
    edge_ids: List[int] = field(default_factory=list)


# ============================================================================
# PART 2: Utility Functions
# ============================================================================

def truncate_ec(ec: str, level: int = 3) -> str:
    """
    截取EC编号到指定层级

    Args:
        ec: 原始EC，如 '1.14.14.130'
        level: 1/2/3，默认3

    Returns:
        截取后的EC，如 '1.14.14' (level=3)

    Examples:
        truncate_ec('1.14.14.130', 3)
        '1.14.14'
        truncate_ec('1.14.14.130', 1)
        '1'
        truncate_ec('1.2.-.-', 2)
        '1.2'
    """
    if not ec or ec == 'unknown':
        return ''

    parts = ec.split('.')
    if len(parts) < level:
        return ''

    result_parts = parts[:level]

    # 过滤通配符
    while result_parts and result_parts[-1] == '-':
        result_parts = result_parts[:-1]

    if len(result_parts) < level:
        return ''

    return '.'.join(result_parts)


def compute_direction_consistency(vectors: np.ndarray) -> float:
    """
    计算一组向量的方向一致性（平均pairwise cosine similarity）

    Args:
        vectors: (n, d) 数组

    Returns:
        平均cosine相似度，范围[-1, 1]
    """
    if len(vectors) < 2:
        return 0.0

    # 过滤零向量
    norms = np.linalg.norm(vectors, axis=1)
    nonzero_mask = norms > 1e-10
    nonzero_vectors = vectors[nonzero_mask]

    if len(nonzero_vectors) < 2:
        return 0.0

    # 归一化
    normalized = nonzero_vectors / np.linalg.norm(nonzero_vectors, axis=1, keepdims=True)

    # 计算pairwise cosine similarity
    # 限制采样数量以提高效率
    n = len(normalized)
    if n > 100:
        indices = np.random.choice(n, 100, replace=False)
        normalized = normalized[indices]
        n = 100

    cos_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_sims.append(np.dot(normalized[i], normalized[j]))

    return np.mean(cos_sims) if cos_sims else 0.0


def compute_dispersion(vectors: np.ndarray) -> float:
    """计算向量组的离散度（到均值的平均距离）"""
    if len(vectors) < 2:
        return 0.0

    mean_vec = np.mean(vectors, axis=0)
    distances = [np.linalg.norm(v - mean_vec) for v in vectors]
    return np.mean(distances)


# ============================================================================
# PART 3: Data Loading
# ============================================================================

class PathwayDataLoader:
    """
    加载路径数据，提取端到端边
    """

    def __init__(self,
                 cache_dir: str = '../kegg_cache',
                 tgsc_file: str = '../02_kegg_mapping/tgsc_to_kegg.csv'):
        self.cache_dir = Path(cache_dir)
        self.tgsc_file = tgsc_file

        # 数据存储
        self.pathways: List[Dict] = []
        self.kegg_to_smiles: Dict[str, str] = {}
        self.kegg_to_odors: Dict[str, Tuple[str, ...]] = {}  # 多标签气味
        self.edges: List[EndToEndEdge] = []

        # 气味属性列表
        self.odor_attributes: List[str] = []

    def load_all(self) -> bool:
        """加载所有数据"""
        print("=" * 80)
        print("Loading Pathway Data")
        print("=" * 80)

        success = True
        success &= self._load_tgsc_data()
        success &= self._load_pathways()

        if success:
            self._extract_end_to_end_edges()

        return success

    def _load_tgsc_data(self) -> bool:
        """加载TGSC数据：KEGG→SMILES映射 + 气味标签"""
        print(f"\nLoading TGSC data from {self.tgsc_file}...")

        try:
            df = pd.read_csv(self.tgsc_file)
            print(f"  Loaded {len(df)} rows")

            # 识别SMILES列
            smiles_cols = ['IsomericSMILES', 'nonStereoSMILES', 'SMILES', 'smiles']
            smiles_col = None
            for col in smiles_cols:
                if col in df.columns:
                    smiles_col = col
                    break

            if smiles_col is None:
                print("  ⚠ No SMILES column found")
                return False

            # 识别气味属性列（二值列，非核心列）
            core_cols = {'TGSC ID', 'KEGG_IDs', 'CID', 'IsomericSMILES',
                         'nonStereoSMILES', 'IUPACName', 'Updated_Desc_v2', 'Solvent'}

            self.odor_attributes = []
            for col in df.columns:
                if col not in core_cols:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 2:
                        if all(v in [0, 1, 0.0, 1.0, True, False] for v in unique_vals):
                            self.odor_attributes.append(col)

            print(f"  Found {len(self.odor_attributes)} odor attributes")

            # 构建映射
            valid_count = 0
            for _, row in df.iterrows():
                kegg_id = row.get('KEGG_IDs', '')
                smiles = row.get(smiles_col, '')

                # 验证KEGG ID格式
                if not pd.notna(kegg_id) or not str(kegg_id).startswith('C'):
                    continue
                if not pd.notna(smiles):
                    continue

                kegg_id = str(kegg_id)
                smiles = str(smiles)

                # SMILES映射
                self.kegg_to_smiles[kegg_id] = smiles

                # 气味标签（多标签）
                odors = []
                for attr in self.odor_attributes:
                    val = row.get(attr, 0)
                    if pd.notna(val) and val == 1:
                        # 排除odorless
                        if 'odorless' not in attr.lower():
                            odors.append(attr)

                if odors:  # 只保留有气味的化合物
                    self.kegg_to_odors[kegg_id] = tuple(sorted(odors))
                    valid_count += 1

            print(f"  ✓ {len(self.kegg_to_smiles)} KEGG→SMILES mappings")
            print(f"  ✓ {len(self.kegg_to_odors)} compounds with odor labels")

            return True

        except FileNotFoundError:
            print(f"  ❌ File not found: {self.tgsc_file}")
            return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    def _load_pathways(self) -> bool:
        """加载路径缓存"""
        cache_file = self.cache_dir / 'pathways_cache.pkl'
        print(f"\nLoading pathways from {cache_file}...")

        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            self.pathways = cache_data.get('pathways', [])
            print(f"  ✓ Loaded {len(self.pathways)} pathways")
            return True

        except FileNotFoundError:
            print(f"  ❌ Cache not found: {cache_file}")
            print("    Run OdorPathwayAnalyzer first")
            return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    def _extract_end_to_end_edges(self):
        """从路径提取端到端边"""
        print("\nExtracting end-to-end edges...")

        edge_count = 0
        skipped_no_smiles = 0
        skipped_no_odor = 0
        skipped_no_ec = 0

        for idx, pw in enumerate(self.pathways):
            source = pw.get('source', '')
            target = pw.get('target', '')
            ec_sequence = pw.get('ec_sequence', [])

            # 检查EC序列
            valid_ecs = [ec for ec in ec_sequence if ec and ec != 'unknown']
            if not valid_ecs:
                skipped_no_ec += 1
                continue

            # 检查SMILES
            source_smiles = self.kegg_to_smiles.get(source)
            target_smiles = self.kegg_to_smiles.get(target)

            if not source_smiles or not target_smiles:
                skipped_no_smiles += 1
                continue

            # 检查气味标签
            source_odors = self.kegg_to_odors.get(source, ())
            target_odors = self.kegg_to_odors.get(target, ())

            if not source_odors or not target_odors:
                skipped_no_odor += 1
                continue

            # 创建边
            edge = EndToEndEdge(
                source_compound=source,
                target_compound=target,
                source_smiles=source_smiles,
                target_smiles=target_smiles,
                source_odors=source_odors,
                target_odors=target_odors,
                ec_sequence=tuple(valid_ecs),
                path_length=len(valid_ecs),
                pathway_id=idx,
                odor_similarity=pw.get('odor_similarity', 0.0)
            )

            self.edges.append(edge)
            edge_count += 1

        print(f"  ✓ Extracted {edge_count} valid edges")
        print(f"  ⊘ Skipped: {skipped_no_smiles} (no SMILES), "
              f"{skipped_no_odor} (no odor), {skipped_no_ec} (no EC)")

        # 统计
        self._print_edge_statistics()

    def _print_edge_statistics(self):
        """打印边的统计信息"""
        if not self.edges:
            return

        print("\n  Edge Statistics:")

        # 路径长度分布
        lengths = Counter(e.path_length for e in self.edges)
        print(f"    Path lengths: {dict(sorted(lengths.items()))}")

        # EC类分布（1位数）
        ec_classes = Counter()
        for e in self.edges:
            for ec in e.ec_sequence:
                ec_class = ec.split('.')[0] if '.' in ec else ec
                ec_classes[ec_class] += 1
        print(f"    EC classes: {dict(sorted(ec_classes.items()))}")

        # 唯一气味对数量
        odor_pairs = set((e.source_odor_set, e.target_odor_set) for e in self.edges)
        print(f"    Unique odor pairs: {len(odor_pairs)}")

        # 重复统计（关键！）
        self._print_duplication_stats()

    def _print_duplication_stats(self):
        """打印重复统计信息"""
        if not self.edges:
            return

        # 统计重复
        src_tgt_ec_keys = [(e.source_compound, e.target_compound, e.ec_sequence) for e in self.edges]
        src_tgt_keys = [(e.source_compound, e.target_compound) for e in self.edges]

        unique_src_tgt_ec = len(set(src_tgt_ec_keys))
        unique_src_tgt = len(set(src_tgt_keys))
        total = len(self.edges)

        print(f"\n  Duplication Statistics (IMPORTANT):")
        print(f"    Total edges: {total}")
        print(f"    Unique (src, tgt, ec_seq): {unique_src_tgt_ec} ({100 * unique_src_tgt_ec / total:.1f}%)")
        print(f"    Unique (src, tgt): {unique_src_tgt} ({100 * unique_src_tgt / total:.1f}%)")
        print(f"    ⚠ Duplication ratio (src,tgt,ec): {100 * (1 - unique_src_tgt_ec / total):.1f}%")
        print(f"    ⚠ Duplication ratio (src,tgt): {100 * (1 - unique_src_tgt / total):.1f}%")

        # 检查是否有严重重复
        if unique_src_tgt_ec < total * 0.9:
            print(f"    ⚠ WARNING: High duplication detected! Consider using deduplicate_edges()")

    def deduplicate_edges(self, mode: str = 'src_tgt_ec') -> int:
        """
        去重边，避免统计膨胀

        Args:
            mode: 去重模式
                - 'src_tgt_ec': 每个(source, target, ec_sequence)保留1条
                - 'src_tgt': 每个(source, target)保留1条（最严格）
                - 'src_tgt_ec_class': 每个(source, target, ec_class_seq)保留1条

        Returns:
            去重后剩余的边数
        """
        if not self.edges:
            return 0

        original_count = len(self.edges)
        seen = set()
        deduplicated = []

        for edge in self.edges:
            if mode == 'src_tgt_ec':
                # 精确EC序列去重
                key = (edge.source_compound, edge.target_compound, edge.ec_sequence)
            elif mode == 'src_tgt':
                # 端点对去重（最严格）
                key = (edge.source_compound, edge.target_compound)
            elif mode == 'src_tgt_ec_class':
                # EC大类序列去重
                ec_classes = tuple(ec.split('.')[0] for ec in edge.ec_sequence if '.' in ec)
                key = (edge.source_compound, edge.target_compound, ec_classes)
            else:
                raise ValueError(f"Unknown dedup mode: {mode}")

            if key not in seen:
                seen.add(key)
                deduplicated.append(edge)

        self.edges = deduplicated
        new_count = len(self.edges)

        print(f"\n  Deduplication (mode='{mode}'):")
        print(f"    Before: {original_count} edges")
        print(f"    After:  {new_count} edges")
        print(f"    Removed: {original_count - new_count} ({100 * (original_count - new_count) / original_count:.1f}%)")

        return new_count


# ============================================================================
# PART 4: Embedding Extractor
# ============================================================================

class EmbeddingExtractor:
    """
    提取分子嵌入向量

    支持三种模式：
    1. OpenPOM中间层embedding（Hook方式，推荐）
    2. OpenPOM预测输出（不推荐，仅用于对比）
    3. RDKit描述符（备用/基线对照）

    关键区别：
    - 中间层embedding：真正的分子结构latent space
    - 预测输出：气味预测空间（138维），会导致循环论证
    """

    # 常见的embedding层名称候选
    EMBEDDING_LAYER_CANDIDATES = [
        'ffn.2',  # 常见：FFN第3层（0-indexed）
        'ffn.4',  # 或第5层
        'ffn_out',
        'embedding_layer',
        'dense.2',
    ]

    def __init__(self, model_dir: str = '../experiments',
                 embedding_dim: int = 256,
                 device: str = 'cpu',
                 embedding_layer_name: str = None,
                 use_prediction_space: bool = False):
        """
        Args:
            model_dir: OpenPOM模型目录
            embedding_dim: 嵌入维度
            device: 计算设备
            embedding_layer_name: 要hook的层名称（None则自动检测）
            use_prediction_space: 是否使用预测输出空间（不推荐，仅对比用）
        """
        self.model_dir = model_dir
        self.embedding_dim = embedding_dim
        self.device = device
        self.embedding_layer_name = embedding_layer_name
        self.use_prediction_space = use_prediction_space

        self.model = None
        self.torch_model = None
        self.featurizer = None
        self._use_real_model = False
        self._hook_handle = None
        self._hooked_output = {}

    def initialize(self):
        """初始化嵌入提取器"""
        print("\nInitializing embedding extractor...")

        if self.use_prediction_space:
            print("  ⚠ WARNING: Using prediction space (not recommended)")
            print("    This validates 'odor prediction space', not 'latent structure space'")

        # 尝试加载OpenPOM模型
        try:
            self._try_load_openpom()
        except Exception as e:
            print(f"  ⚠ OpenPOM not available: {e}")
            print("  Using RDKit descriptors as fallback")
            self._use_real_model = False

    def _try_load_openpom(self):
        """尝试加载OpenPOM模型并设置hook"""
        import deepchem as dc
        from openpom.models.mpnn_pom import MPNNPOMModel
        from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants

        checkpoint = Path(self.model_dir) / 'checkpoint1.pt'
        if not checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint at {checkpoint}")

        # 初始化模型
        self.model = MPNNPOMModel(
            n_tasks=138,
            batch_size=128,
            learning_rate=0.0001,
            node_out_feats=100,
            edge_hidden_feats=75,
            edge_out_feats=100,
            num_step_message_passing=5,
            mpnn_residual=True,
            message_aggregator_type='sum',
            mode='classification',
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,
            n_classes=1,
            readout_type='set2set',
            num_step_set2set=3,
            num_layer_set2set=2,
            ffn_hidden_list=[392, 392],
            ffn_embeddings=self.embedding_dim,
            ffn_activation='relu',
            ffn_dropout_p=0.12,
            model_dir=self.model_dir,
            device_name=self.device
        )
        self.model.restore()
        self.featurizer = GraphFeaturizer()
        self.torch_model = self.model.model

        # 设置hook（如果不用prediction space）
        if not self.use_prediction_space:
            self._setup_hook()

        self._use_real_model = True
        print(f"  ✓ Loaded OpenPOM model from {self.model_dir}")

    def _setup_hook(self):
        """设置forward hook来捕获中间层输出"""
        import torch

        # 如果指定了层名称，直接使用
        if self.embedding_layer_name:
            target_layer = self._get_layer_by_name(self.embedding_layer_name)
            if target_layer is not None:
                self._register_hook(target_layer, self.embedding_layer_name)
                return

        # 自动检测：找到输出256维的层
        found = False
        for name, module in self.torch_model.named_modules():
            if hasattr(module, 'out_features') and module.out_features == self.embedding_dim:
                # 找到了256维输出的层
                print(f"  ✓ Auto-detected embedding layer: {name} ({module.in_features}→{module.out_features})")
                self._register_hook(module, name)
                self.embedding_layer_name = name
                found = True
                break

        if not found:
            # 尝试候选列表
            for candidate in self.EMBEDDING_LAYER_CANDIDATES:
                layer = self._get_layer_by_name(candidate)
                if layer is not None:
                    print(f"  ✓ Found embedding layer: {candidate}")
                    self._register_hook(layer, candidate)
                    self.embedding_layer_name = candidate
                    found = True
                    break

        if not found:
            print("  ⚠ Could not find embedding layer, falling back to prediction space")
            print("    Run inspect_openpom_structure.py to find the correct layer name")
            self.use_prediction_space = True

    def _get_layer_by_name(self, name: str):
        """通过名称获取层"""
        try:
            parts = name.split('.')
            module = self.torch_model
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            return module
        except (AttributeError, IndexError, KeyError):
            return None

    def _register_hook(self, layer, name: str):
        """注册forward hook"""

        def hook_fn(module, input, output):
            # 保存输出（detach以避免计算图问题）
            if hasattr(output, 'detach'):
                self._hooked_output['embedding'] = output.detach().cpu().numpy()
            else:
                self._hooked_output['embedding'] = np.array(output)

        self._hook_handle = layer.register_forward_hook(hook_fn)
        print(f"  ✓ Registered hook on layer: {name}")

    def get_embedding(self, smiles: str) -> Optional[np.ndarray]:
        """获取单个SMILES的嵌入"""
        if self._use_real_model:
            return self._get_openpom_embedding(smiles)
        else:
            return self._get_rdkit_embedding(smiles)

    def _get_openpom_embedding(self, smiles: str) -> Optional[np.ndarray]:
        """从OpenPOM获取嵌入（通过hook或prediction）"""
        try:
            import deepchem as dc

            featurized = self.featurizer.featurize([smiles])
            if featurized is None or len(featurized) == 0:
                return None

            dataset = dc.data.NumpyDataset(featurized)

            # Forward pass（会触发hook）
            self._hooked_output.clear()
            predictions = self.model.predict(dataset)

            if self.use_prediction_space:
                # 使用预测输出（不推荐）
                embedding = np.zeros(self.embedding_dim)
                pred_flat = predictions.flatten()
                embedding[:len(pred_flat)] = pred_flat
                return embedding
            else:
                # 使用hook捕获的中间层（推荐）
                if 'embedding' in self._hooked_output:
                    emb = self._hooked_output['embedding'].flatten()
                    # 确保维度正确
                    if len(emb) >= self.embedding_dim:
                        return emb[:self.embedding_dim]
                    else:
                        # padding
                        result = np.zeros(self.embedding_dim)
                        result[:len(emb)] = emb
                        return result
                else:
                    # Hook失败，fallback
                    print(f"    ⚠ Hook failed for {smiles[:20]}..., using prediction")
                    embedding = np.zeros(self.embedding_dim)
                    pred_flat = predictions.flatten()
                    embedding[:len(pred_flat)] = pred_flat
                    return embedding

        except Exception as e:
            return None

    def cleanup(self):
        """清理hook"""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self):
        self.cleanup()

    def _get_rdkit_embedding(self, smiles: str) -> Optional[np.ndarray]:
        """使用RDKit描述符生成伪嵌入"""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            features = []

            # 物理化学描述符（约20个）
            features.append(Descriptors.MolWt(mol) / 500)
            features.append(Descriptors.MolLogP(mol) / 10)
            features.append(Descriptors.TPSA(mol) / 200)
            features.append(rdMolDescriptors.CalcNumRotatableBonds(mol) / 20)
            features.append(rdMolDescriptors.CalcNumHBD(mol) / 10)
            features.append(rdMolDescriptors.CalcNumHBA(mol) / 10)
            features.append(rdMolDescriptors.CalcNumRings(mol) / 5)
            features.append(rdMolDescriptors.CalcNumAromaticRings(mol) / 5)
            features.append(rdMolDescriptors.CalcFractionCSP3(mol))
            features.append(mol.GetNumAtoms() / 50)
            features.append(mol.GetNumBonds() / 50)
            features.append(Descriptors.NumValenceElectrons(mol) / 100)
            features.append(Descriptors.NumRadicalElectrons(mol))
            features.append(rdMolDescriptors.CalcNumAliphaticRings(mol) / 5)
            features.append(rdMolDescriptors.CalcNumSaturatedRings(mol) / 5)
            features.append(rdMolDescriptors.CalcNumHeterocycles(mol) / 5)
            features.append(Descriptors.FpDensityMorgan1(mol))
            features.append(Descriptors.FpDensityMorgan2(mol))
            features.append(Descriptors.FpDensityMorgan3(mol))
            features.append(Descriptors.HeavyAtomMolWt(mol) / 500)

            # Morgan指纹（剩余维度）
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=self.embedding_dim - len(features))
            morgan_array = np.array(morgan_fp)

            # 组合
            embedding = np.concatenate([np.array(features), morgan_array.astype(float)])

            return embedding[:self.embedding_dim]

        except ImportError:
            # 无RDKit，使用hash
            hash_val = hash(smiles) % (2 ** 32)
            np.random.seed(hash_val)
            return np.random.randn(self.embedding_dim) * 0.1
        except Exception:
            return None

    def get_batch_embeddings(self, smiles_list: List[str],
                             show_progress: bool = True) -> Dict[str, np.ndarray]:
        """批量获取嵌入"""
        embeddings = {}
        iterator = tqdm(smiles_list, desc="Extracting embeddings") if show_progress else smiles_list

        for smiles in iterator:
            emb = self.get_embedding(smiles)
            if emb is not None:
                embeddings[smiles] = emb

        return embeddings


# ============================================================================
# PART 5: Grouping Strategies
# ============================================================================

class GroupingStrategy:
    """
    分组策略基类
    """
    name: str = "base"

    def get_group_key(self, edge: EndToEndEdge, ec_level: int = 3) -> Any:
        """返回边的分组键"""
        raise NotImplementedError


class ECSequenceGrouping(GroupingStrategy):
    """按EC序列分组"""
    name = "ec_sequence"

    def get_group_key(self, edge: EndToEndEdge, ec_level: int = 3) -> Tuple[str, ...]:
        return edge.get_ec_at_level(ec_level)


class OdorPairGrouping(GroupingStrategy):
    """按气味对分组（源气味集合, 目标气味集合）"""
    name = "odor_pair"

    def get_group_key(self, edge: EndToEndEdge, ec_level: int = 3) -> Tuple[FrozenSet, FrozenSet]:
        return (edge.source_odor_set, edge.target_odor_set)


class SourceOdorFirstECGrouping(GroupingStrategy):
    """按(源气味, 第一个EC)分组"""
    name = "source_odor_first_ec"

    def get_group_key(self, edge: EndToEndEdge, ec_level: int = 3) -> Tuple:
        first_ec = truncate_ec(edge.first_ec, ec_level)
        # 为每个源气味创建一个分组
        return tuple((odor, first_ec) for odor in sorted(edge.source_odors))


class TargetOdorLastECGrouping(GroupingStrategy):
    """按(目标气味, 最后一个EC)分组"""
    name = "target_odor_last_ec"

    def get_group_key(self, edge: EndToEndEdge, ec_level: int = 3) -> Tuple:
        last_ec = truncate_ec(edge.last_ec, ec_level)
        return tuple((odor, last_ec) for odor in sorted(edge.target_odors))


class FullCombinationGrouping(GroupingStrategy):
    """按(源气味, first_ec, 目标气味, last_ec)完整组合分组"""
    name = "full_combination"

    def get_group_key(self, edge: EndToEndEdge, ec_level: int = 3) -> Tuple:
        first_ec = truncate_ec(edge.first_ec, ec_level)
        last_ec = truncate_ec(edge.last_ec, ec_level)
        # 展开为多个key（笛卡尔积）
        keys = []
        for s_odor in edge.source_odors:
            for t_odor in edge.target_odors:
                keys.append((s_odor, first_ec, t_odor, last_ec))
        return tuple(keys)


class SingleSourceOdorFirstECGrouping(GroupingStrategy):
    """按单个(源气味, 第一个EC)分组 - 展开多标签"""
    name = "single_source_first_ec"

    def get_group_key(self, edge: EndToEndEdge, ec_level: int = 3) -> List[Tuple[str, str]]:
        first_ec = truncate_ec(edge.first_ec, ec_level)
        return [(odor, first_ec) for odor in edge.source_odors]


class SingleTargetOdorLastECGrouping(GroupingStrategy):
    """按单个(目标气味, 最后一个EC)分组 - 展开多标签"""
    name = "single_target_last_ec"

    def get_group_key(self, edge: EndToEndEdge, ec_level: int = 3) -> List[Tuple[str, str]]:
        last_ec = truncate_ec(edge.last_ec, ec_level)
        return [(odor, last_ec) for odor in edge.target_odors]


class FirstECOnlyGrouping(GroupingStrategy):
    """只按第一个EC分组（纯EC策略，不含气味信息）"""
    name = "first_ec_only"

    def get_group_key(self, edge: EndToEndEdge, ec_level: int = 3) -> Tuple[str]:
        first_ec = truncate_ec(edge.first_ec, ec_level)
        if first_ec:
            return (first_ec,)
        return None


class LastECOnlyGrouping(GroupingStrategy):
    """只按最后一个EC分组（纯EC策略，不含气味信息）"""
    name = "last_ec_only"

    def get_group_key(self, edge: EndToEndEdge, ec_level: int = 3) -> Tuple[str]:
        last_ec = truncate_ec(edge.last_ec, ec_level)
        if last_ec:
            return (last_ec,)
        return None


class ECClassSequenceGrouping(GroupingStrategy):
    """按EC大类序列分组（只看EC第1位）"""
    name = "ec_class_sequence"

    def get_group_key(self, edge: EndToEndEdge, ec_level: int = 3) -> Tuple[str, ...]:
        # 强制只取第1位
        classes = tuple(truncate_ec(ec, 1) for ec in edge.ec_sequence
                        if truncate_ec(ec, 1))
        return classes if classes else None


# 所有可用策略
GROUPING_STRATEGIES = {
    'ec_sequence': ECSequenceGrouping(),
    'first_ec_only': FirstECOnlyGrouping(),  # 新增：纯EC，不含气味
    'last_ec_only': LastECOnlyGrouping(),  # 新增：纯EC，不含气味
    'ec_class_sequence': ECClassSequenceGrouping(),  # 新增：EC大类序列
    'odor_pair': OdorPairGrouping(),
    'source_odor_first_ec': SingleSourceOdorFirstECGrouping(),
    'target_odor_last_ec': SingleTargetOdorLastECGrouping(),
    'full_combination': FullCombinationGrouping(),
}

DEFAULT_STRATEGIES = [
    'ec_sequence',
    'first_ec_only',
    'last_ec_only',
    'ec_class_sequence',
]


# ============================================================================
# PART 6: Delta Analyzer
# ============================================================================

class EndToEndDeltaAnalyzer:
    """
    端到端路径的Delta分析器
    """

    def __init__(self,
                 edges: List[EndToEndEdge],
                 embeddings: Dict[str, np.ndarray],
                 ec_level: int = 3):
        """
        Args:
            edges: 端到端边列表
            embeddings: SMILES→嵌入映射
            ec_level: EC截取层级 (1/2/3)
        """
        self.edges = edges
        self.embeddings = embeddings
        self.ec_level = ec_level

        # 计算的数据
        self.deltas: List[Tuple[int, np.ndarray]] = []  # (edge_idx, delta_vector)
        self.group_stats: Dict[str, Dict[Any, GroupStatistics]] = {}

    def compute_deltas(self):
        """计算所有边的Delta向量"""
        print("\n" + "=" * 80)
        print(f"Computing Deltas (EC level = {self.ec_level})")
        print("=" * 80)

        valid_count = 0
        for idx, edge in enumerate(self.edges):
            source_emb = self.embeddings.get(edge.source_smiles)
            target_emb = self.embeddings.get(edge.target_smiles)

            if source_emb is None or target_emb is None:
                continue

            delta = target_emb - source_emb
            self.deltas.append((idx, delta))
            valid_count += 1

        print(f"  ✓ Computed {valid_count} delta vectors")

    def analyze_by_strategy(self, strategy_name: str, min_group_size: int = 3) -> Dict[Any, GroupStatistics]:
        """
        按指定策略分组并分析

        Args:
            strategy_name: 分组策略名称
            min_group_size: 最小分组大小

        Returns:
            分组键→统计信息的映射
        """
        if strategy_name not in GROUPING_STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy = GROUPING_STRATEGIES[strategy_name]
        print(f"\n  Analyzing by '{strategy_name}'...")

        # 分组
        groups = defaultdict(list)  # key → [(edge_idx, delta)]

        for edge_idx, delta in self.deltas:
            edge = self.edges[edge_idx]
            keys = strategy.get_group_key(edge, self.ec_level)

            # 处理返回多个key的情况（多标签展开）
            if isinstance(keys, list):
                for key in keys:
                    if key is None:
                        continue
                    groups[key].append((edge_idx, delta))
            else:
                if keys is None:
                    continue
                groups[keys].append((edge_idx, delta))

        # 过滤小组
        filtered_groups = {k: v for k, v in groups.items() if len(v) >= min_group_size}

        print(f"    Total groups: {len(groups)}")
        print(f"    Groups with ≥{min_group_size} edges: {len(filtered_groups)}")

        # 计算统计
        stats = {}
        for key, edge_deltas in filtered_groups.items():
            edge_ids = [ed[0] for ed in edge_deltas]
            delta_vectors = np.array([ed[1] for ed in edge_deltas])

            mean_delta = np.mean(delta_vectors, axis=0)
            dispersion = compute_dispersion(delta_vectors)
            consistency = compute_direction_consistency(delta_vectors)
            mean_magnitude = np.mean([np.linalg.norm(d) for d in delta_vectors])

            stats[key] = GroupStatistics(
                group_key=key,
                n_edges=len(edge_deltas),
                delta_vectors=delta_vectors,
                mean_delta=mean_delta,
                delta_dispersion=dispersion,
                direction_consistency=consistency,
                mean_magnitude=mean_magnitude,
                edge_ids=edge_ids
            )

        self.group_stats[strategy_name] = stats

        # 打印摘要（包括加权和非加权）
        if stats:
            consistencies = [s.direction_consistency for s in stats.values()]
            n_edges_list = [s.n_edges for s in stats.values()]

            # 非加权平均
            unweighted_mean = np.mean(consistencies)

            # 加权平均（按边数）
            total_edges = sum(n_edges_list)
            weighted_mean = sum(c * n / total_edges for c, n in zip(consistencies, n_edges_list))

            print(f"    Consistency (unweighted): mean={unweighted_mean:.4f}, std={np.std(consistencies):.4f}")
            print(f"    Consistency (weighted):   mean={weighted_mean:.4f}")
            print(f"    Total edges in groups: {total_edges}")

        return stats

    def run_shuffle_test(self, strategy_name: str, n_permutations: int = 1000,
                         shuffle_mode: str = 'global',
                         max_samples: int = 50000) -> Dict:
        """
        置换检验：验证分组的非随机性

        Args:
            strategy_name: 分组策略名称
            n_permutations: 置换次数
            shuffle_mode: 置换模式
                - 'global': 完全随机打乱（当前默认）
                - 'within_source': 只在同source内打乱target
                - 'within_odor_pair': 只在同(source_odors, target_odors)内打乱EC标签
                - 'within_length': 只在同path_length内打乱
            max_samples: 最大样本数（超过则采样，加速计算）

        Returns:
            包含统计结果的字典
        """
        print(f"\n  Running shuffle test for '{strategy_name}' (mode={shuffle_mode})...")

        if strategy_name not in self.group_stats:
            self.analyze_by_strategy(strategy_name)

        stats = self.group_stats[strategy_name]
        if not stats:
            print("    ⚠ No groups to test")
            return {}

        # 真实的一致性（加权和非加权）
        consistencies = [s.direction_consistency for s in stats.values()]
        n_edges_list = [s.n_edges for s in stats.values()]
        total_edges = sum(n_edges_list)

        real_mean_unweighted = np.mean(consistencies)
        real_mean_weighted = sum(c * n / total_edges for c, n in zip(consistencies, n_edges_list))

        # 收集所有delta向量和原始分组索引
        all_deltas = []
        all_key_indices = []
        all_edge_indices = []  # 用于条件置换

        # 建立键到索引的映射
        unique_keys = list(stats.keys())
        key_to_idx = {k: i for i, k in enumerate(unique_keys)}

        for key, stat in stats.items():
            key_idx = key_to_idx[key]
            for i, delta in enumerate(stat.delta_vectors):
                all_deltas.append(delta)
                all_key_indices.append(key_idx)
                # 保存edge索引用于条件置换
                if i < len(stat.edge_ids):
                    all_edge_indices.append(stat.edge_ids[i])
                else:
                    all_edge_indices.append(-1)

        all_deltas = np.array(all_deltas)
        all_key_indices = np.array(all_key_indices)

        n_samples = len(all_deltas)
        n_keys = len(unique_keys)

        # 如果样本太多，进行采样
        sampled = False
        if n_samples > max_samples:
            print(f"    ⚠ Too many samples ({n_samples}), sampling {max_samples} for speed")
            sample_idx = np.random.choice(n_samples, max_samples, replace=False)
            all_deltas = all_deltas[sample_idx]
            all_key_indices = all_key_indices[sample_idx]
            all_edge_indices = [all_edge_indices[i] for i in sample_idx]
            n_samples = max_samples
            sampled = True

        print(f"    Total samples: {n_samples}, Groups: {n_keys}")

        # 构建条件置换的约束组
        constraint_groups = self._build_constraint_groups(all_edge_indices, shuffle_mode)

        if shuffle_mode != 'global':
            n_constraint_groups = len(set(constraint_groups))
            print(f"    Constraint groups ({shuffle_mode}): {n_constraint_groups}")

        # 置换测试
        shuffled_unweighted = []
        shuffled_weighted = []

        for _ in tqdm(range(n_permutations), desc="    Shuffling"):
            # 根据模式进行置换
            if shuffle_mode == 'global':
                shuffled_indices = np.random.permutation(all_key_indices)
            else:
                # 条件置换：只在同一约束组内打乱
                shuffled_indices = self._conditional_shuffle(all_key_indices, constraint_groups)

            # 重新计算每组一致性
            shuffled_groups = defaultdict(list)
            for delta, key_idx in zip(all_deltas, shuffled_indices):
                shuffled_groups[key_idx].append(delta)

            cons_list = []
            n_list = []
            for key_idx, deltas in shuffled_groups.items():
                if len(deltas) >= 3:
                    cons = compute_direction_consistency(np.array(deltas))
                    cons_list.append(cons)
                    n_list.append(len(deltas))

            if cons_list:
                shuffled_unweighted.append(np.mean(cons_list))
                total_n = sum(n_list)
                shuffled_weighted.append(sum(c * n / total_n for c, n in zip(cons_list, n_list)))

        shuffled_unweighted = np.array(shuffled_unweighted)
        shuffled_weighted = np.array(shuffled_weighted)

        # 检测无效检验组合（分组键 == 约束键）
        invalid_test = False
        if shuffle_mode == 'within_odor_pair' and strategy_name == 'odor_pair':
            print(f"    ⚠ WARNING: Invalid test combination!")
            print(
                f"      Strategy '{strategy_name}' groups by odor_pair, shuffle '{shuffle_mode}' constrains by odor_pair")
            print(f"      → Shuffling within the same grouping key has no effect")
            invalid_test = True

        if shuffle_mode == 'within_source' and strategy_name == 'source_odor_first_ec':
            print(f"    ⚠ Note: Partial overlap between grouping key and shuffle constraint")

        # 计算p值和效应量（非加权）- 添加数值稳定性
        p_value_unweighted = np.mean(shuffled_unweighted >= real_mean_unweighted)
        std_unweighted = np.std(shuffled_unweighted)
        # 设置 std 下限避免除零爆炸
        MIN_STD = 1e-6
        if std_unweighted > MIN_STD:
            effect_size_unweighted = (real_mean_unweighted - np.mean(shuffled_unweighted)) / std_unweighted
        else:
            # std 太小，用差值代替（并标记）
            effect_size_unweighted = float('inf') if real_mean_unweighted > np.mean(shuffled_unweighted) else 0.0

        # 计算p值和效应量（加权）
        p_value_weighted = np.mean(shuffled_weighted >= real_mean_weighted)
        std_weighted = np.std(shuffled_weighted)
        if std_weighted > MIN_STD:
            effect_size_weighted = (real_mean_weighted - np.mean(shuffled_weighted)) / std_weighted
        else:
            effect_size_weighted = float('inf') if real_mean_weighted > np.mean(shuffled_weighted) else 0.0

        # 使用更保守的p值计算 (k+1)/(N+1)
        k_unweighted = np.sum(shuffled_unweighted >= real_mean_unweighted)
        k_weighted = np.sum(shuffled_weighted >= real_mean_weighted)
        p_value_unweighted_conservative = (k_unweighted + 1) / (n_permutations + 1)
        p_value_weighted_conservative = (k_weighted + 1) / (n_permutations + 1)

        results = {
            'strategy': strategy_name,
            'shuffle_mode': shuffle_mode,
            'sampled': sampled,
            'n_samples_used': n_samples,
            'invalid_test': invalid_test,  # 标记无效检验
            # 非加权结果
            'real_mean_consistency': real_mean_unweighted,
            'shuffled_mean': np.mean(shuffled_unweighted),
            'shuffled_std': std_unweighted,
            'real_null_delta': real_mean_unweighted - np.mean(shuffled_unweighted),  # 关键指标
            'p_value': p_value_unweighted,
            'p_value_conservative': p_value_unweighted_conservative,
            'effect_size': effect_size_unweighted if effect_size_unweighted != float('inf') else 9999.0,
            # 加权结果
            'real_mean_weighted': real_mean_weighted,
            'shuffled_mean_weighted': np.mean(shuffled_weighted),
            'shuffled_std_weighted': std_weighted,
            'real_null_delta_weighted': real_mean_weighted - np.mean(shuffled_weighted),
            'p_value_weighted': p_value_weighted,
            'p_value_weighted_conservative': p_value_weighted_conservative,
            'effect_size_weighted': effect_size_weighted if effect_size_weighted != float('inf') else 9999.0,
            # 元信息
            'n_groups': len(stats),
            'n_samples': n_samples,
            'shuffled_distribution': shuffled_unweighted,
            'shuffled_distribution_weighted': shuffled_weighted
        }

        sample_note = " [SAMPLED]" if sampled else ""
        invalid_note = " [INVALID TEST]" if invalid_test else ""
        print(
            f"    [Unweighted] Real: {real_mean_unweighted:.4f}, Null: {np.mean(shuffled_unweighted):.4f} ± {std_unweighted:.4f}, Δ={real_mean_unweighted - np.mean(shuffled_unweighted):.4f}{sample_note}{invalid_note}")
        print(
            f"                 p={p_value_unweighted_conservative:.4f} (conservative), effect={effect_size_unweighted:.2f}" if effect_size_unweighted != float(
                'inf') else f"                 p={p_value_unweighted_conservative:.4f} (conservative), effect=INF (std≈0)")
        print(
            f"    [Weighted]   Real: {real_mean_weighted:.4f}, Null: {np.mean(shuffled_weighted):.4f} ± {std_weighted:.4f}, Δ={real_mean_weighted - np.mean(shuffled_weighted):.4f}")
        print(
            f"                 p={p_value_weighted_conservative:.4f} (conservative), effect={effect_size_weighted:.2f}" if effect_size_weighted != float(
                'inf') else f"                 p={p_value_weighted_conservative:.4f} (conservative), effect=INF (std≈0)")

        return results

    def _build_constraint_groups(self, edge_indices: List[int], shuffle_mode: str) -> np.ndarray:
        """
        构建条件置换的约束组

        返回一个数组，相同值表示在同一个约束组内
        """
        n = len(edge_indices)

        if shuffle_mode == 'global':
            # 全局打乱：所有样本在同一组
            return np.zeros(n, dtype=int)

        constraint_groups = np.zeros(n, dtype=int)

        if shuffle_mode == 'within_source':
            # 按source分组
            source_to_group = {}
            group_id = 0
            for i, edge_idx in enumerate(edge_indices):
                if edge_idx >= 0 and edge_idx < len(self.edges):
                    source = self.edges[edge_idx].source_compound
                    if source not in source_to_group:
                        source_to_group[source] = group_id
                        group_id += 1
                    constraint_groups[i] = source_to_group[source]
                else:
                    constraint_groups[i] = -1  # 无效边单独一组

        elif shuffle_mode == 'within_odor_pair':
            # 按(source_odors, target_odors)分组
            odor_pair_to_group = {}
            group_id = 0
            for i, edge_idx in enumerate(edge_indices):
                if edge_idx >= 0 and edge_idx < len(self.edges):
                    edge = self.edges[edge_idx]
                    odor_pair = (edge.source_odor_set, edge.target_odor_set)
                    if odor_pair not in odor_pair_to_group:
                        odor_pair_to_group[odor_pair] = group_id
                        group_id += 1
                    constraint_groups[i] = odor_pair_to_group[odor_pair]
                else:
                    constraint_groups[i] = -1

        elif shuffle_mode == 'within_length':
            # 按path_length分组
            for i, edge_idx in enumerate(edge_indices):
                if edge_idx >= 0 and edge_idx < len(self.edges):
                    constraint_groups[i] = self.edges[edge_idx].path_length
                else:
                    constraint_groups[i] = -1
        else:
            raise ValueError(f"Unknown shuffle_mode: {shuffle_mode}")

        return constraint_groups

    def _conditional_shuffle(self, key_indices: np.ndarray, constraint_groups: np.ndarray) -> np.ndarray:
        """
        条件置换：只在同一约束组内打乱
        """
        shuffled = key_indices.copy()
        unique_groups = np.unique(constraint_groups)

        for group in unique_groups:
            if group < 0:  # 跳过无效组
                continue
            mask = constraint_groups == group
            indices = np.where(mask)[0]
            if len(indices) > 1:
                # 在组内打乱
                shuffled_values = key_indices[indices].copy()
                np.random.shuffle(shuffled_values)
                shuffled[indices] = shuffled_values

        return shuffled

    def compute_between_group_separation(self, strategy_name: str, max_groups: int = 500) -> Dict:
        """
        计算组间分离度

        Args:
            strategy_name: 策略名称
            max_groups: 最大组数（超过则采样，避免O(n²)爆炸）
        """
        if strategy_name not in self.group_stats:
            self.analyze_by_strategy(strategy_name)

        stats = self.group_stats[strategy_name]
        if len(stats) < 2:
            return {'error': 'Need at least 2 groups'}

        keys = list(stats.keys())
        n_groups = len(keys)

        # 如果组数太多，采样以避免O(n²)爆炸
        sampled = False
        if n_groups > max_groups:
            print(f"    ⚠ Too many groups ({n_groups}) for separation matrix, sampling {max_groups}")
            sample_indices = np.random.choice(n_groups, max_groups, replace=False)
            keys = [keys[i] for i in sample_indices]
            n_groups = max_groups
            sampled = True

        mean_deltas = np.array([stats[k].mean_delta for k in keys])

        # 向量化计算余弦距离（避免双重循环）
        print(f"    Computing separation ({n_groups} groups)...")
        norms = np.linalg.norm(mean_deltas, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1  # 避免除零
        normalized = mean_deltas / norms
        cos_sim = np.dot(normalized, normalized.T)
        cos_dist = 1 - cos_sim

        # 欧氏距离矩阵
        euc_dist = squareform(pdist(mean_deltas))

        # 平均分离度（只取上三角）
        upper_idx = np.triu_indices(n_groups, k=1)

        return {
            'strategy': strategy_name,
            'n_groups': n_groups,
            'sampled': sampled,
            'avg_cosine_separation': float(np.mean(cos_dist[upper_idx])),
            'avg_euclidean_separation': float(np.mean(euc_dist[upper_idx])),
            'std_cosine_separation': float(np.std(cos_dist[upper_idx])),
            'std_euclidean_separation': float(np.std(euc_dist[upper_idx]))
        }


# ============================================================================
# PART 7: Visualization
# ============================================================================

class ValidationVisualizer:
    """验证结果可视化"""

    def __init__(self, output_dir: str = './ec_validation_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_shuffle_test(self, results: Dict, strategy_name: str):
        """绘制置换检验结果"""
        fig, ax = plt.subplots(figsize=(10, 6))

        shuffled = results.get('shuffled_distribution', [])
        if len(shuffled) == 0:
            return

        ax.hist(shuffled, bins=50, alpha=0.7, color='steelblue',
                edgecolor='black', label='Null distribution')

        real_val = results['real_mean_consistency']
        ax.axvline(real_val, color='red', linewidth=3, linestyle='--',
                   label=f'Real: {real_val:.4f}')

        # 使用conservative p-value
        p_val = results.get('p_value_conservative', results.get('p_value', 0))
        effect = results.get('effect_size', 0)
        shuffle_mode = results.get('shuffle_mode', 'global')

        ax.set_xlabel('Mean Direction Consistency', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f"Shuffle Test: {strategy_name}\n"
                     f"mode={shuffle_mode}, p={p_val:.4f}, effect={effect:.2f}",
                     fontsize=14, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / f'shuffle_test_{strategy_name}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: shuffle_test_{strategy_name}.png")

    def plot_group_consistency_distribution(self, stats: Dict[Any, GroupStatistics],
                                            strategy_name: str):
        """绘制分组一致性分布"""
        if not stats:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        consistencies = [s.direction_consistency for s in stats.values()]
        n_edges = [s.n_edges for s in stats.values()]

        # 左：直方图
        ax1.hist(consistencies, bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black')
        ax1.axvline(np.mean(consistencies), color='red', linestyle='--', lw=2,
                    label=f'Mean: {np.mean(consistencies):.3f}')
        ax1.set_xlabel('Direction Consistency', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Groups', fontsize=12, fontweight='bold')
        ax1.set_title(f'Consistency Distribution ({strategy_name})', fontsize=13, fontweight='bold')
        ax1.legend()

        # 右：一致性 vs 样本量
        ax2.scatter(n_edges, consistencies, alpha=0.6, c='coral', edgecolors='black', s=50)
        ax2.set_xlabel('Number of Edges', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Direction Consistency', fontsize=12, fontweight='bold')
        ax2.set_title('Consistency vs Sample Size', fontsize=13, fontweight='bold')

        if len(consistencies) > 2:
            try:
                from scipy.stats import pearsonr as scipy_pearsonr
                corr, pval = scipy_pearsonr(n_edges, consistencies)
                ax2.text(0.95, 0.05, f'r={corr:.3f}, p={pval:.4f}',
                         transform=ax2.transAxes, fontsize=10,
                         va='bottom', ha='right',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            except Exception:
                pass

        plt.tight_layout()
        plt.savefig(self.output_dir / f'consistency_dist_{strategy_name}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: consistency_dist_{strategy_name}.png")

    def plot_top_groups(self, stats: Dict[Any, GroupStatistics], strategy_name: str, top_k: int = 20):
        """绘制Top分组"""
        if not stats:
            return

        # 按一致性排序
        sorted_groups = sorted(stats.items(),
                               key=lambda x: x[1].direction_consistency,
                               reverse=True)[:top_k]

        fig, ax = plt.subplots(figsize=(12, 8))

        keys = [str(g[0])[:50] for g in sorted_groups]  # 截断长key
        consistencies = [g[1].direction_consistency for g in sorted_groups]
        n_edges = [g[1].n_edges for g in sorted_groups]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(keys)))
        bars = ax.barh(range(len(keys)), consistencies, color=colors, edgecolor='black')

        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys, fontsize=9)
        ax.invert_yaxis()

        # 标注样本量
        for i, (bar, n) in enumerate(zip(bars, n_edges)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'n={n}', va='center', fontsize=8)

        ax.set_xlabel('Direction Consistency', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_k} Groups by Consistency ({strategy_name})',
                     fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'top_groups_{strategy_name}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: top_groups_{strategy_name}.png")

    def plot_pca_with_arrows(self, analyzer: EndToEndDeltaAnalyzer,
                             strategy_name: str, top_k_groups: int = 8):
        """绘制PCA空间中的转化箭头"""
        print(f"\n  Generating PCA visualization for '{strategy_name}'...")

        if strategy_name not in analyzer.group_stats:
            print("    ⚠ No stats for this strategy")
            return

        stats = analyzer.group_stats[strategy_name]
        if not stats:
            return

        # 收集所有嵌入点
        all_points = []
        all_labels = []

        for edge_idx, delta in analyzer.deltas:
            edge = analyzer.edges[edge_idx]
            source_emb = analyzer.embeddings.get(edge.source_smiles)
            target_emb = analyzer.embeddings.get(edge.target_smiles)

            if source_emb is not None:
                all_points.append(source_emb)
                all_labels.append(edge.source_compound)
            if target_emb is not None:
                all_points.append(target_emb)
                all_labels.append(edge.target_compound)

        if len(all_points) < 10:
            print("    ⚠ Not enough points for PCA")
            return

        # PCA
        all_points = np.array(all_points)
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(all_points)

        # 建立映射
        label_to_pca = {}
        for i, label in enumerate(all_labels):
            label_to_pca[label] = pca_coords[i]

        fig, ax = plt.subplots(figsize=(14, 10))

        # 绘制所有点
        ax.scatter(pca_coords[:, 0], pca_coords[:, 1], alpha=0.2, s=15, c='lightgray')

        # 选择Top分组
        top_groups = sorted(stats.items(),
                            key=lambda x: x[1].n_edges,
                            reverse=True)[:top_k_groups]

        colors = plt.cm.tab10(np.linspace(0, 1, len(top_groups)))

        # 为每个分组绘制箭头
        for (key, stat), color in zip(top_groups, colors):
            # 采样一些边
            sample_size = min(15, len(stat.edge_ids))
            sampled_ids = np.random.choice(stat.edge_ids, sample_size, replace=False)

            for edge_idx in sampled_ids:
                edge = analyzer.edges[edge_idx]

                if edge.source_compound in label_to_pca and edge.target_compound in label_to_pca:
                    start = label_to_pca[edge.source_compound]
                    end = label_to_pca[edge.target_compound]

                    ax.annotate('', xy=end, xytext=start,
                                arrowprops=dict(arrowstyle='->', color=color,
                                                alpha=0.5, lw=1.2))

        # 图例
        legend_patches = [mpatches.Patch(color=colors[i], label=f'{str(k)[:30]}...' if len(str(k)) > 30 else str(k))
                          for i, (k, _) in enumerate(top_groups)]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Transformation Arrows in PCA Space\n({strategy_name})',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'pca_arrows_{strategy_name}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: pca_arrows_{strategy_name}.png")

    def plot_summary_dashboard(self, all_results: Dict, shuffle_mode: str = 'global'):
        """绘制总结仪表板"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # 提取指定shuffle_mode的结果
        strategies = list(all_results.get('shuffle_tests', {}).keys())

        def get_shuffle_result(strategy, key, default=None):
            """从嵌套结构中获取结果"""
            shuffle_tests = all_results.get('shuffle_tests', {}).get(strategy, {})
            if isinstance(shuffle_tests, dict) and shuffle_mode in shuffle_tests:
                return shuffle_tests[shuffle_mode].get(key, default)
            elif isinstance(shuffle_tests, dict) and key in shuffle_tests:
                # 兼容旧格式
                return shuffle_tests.get(key, default)
            return default

        # 1. 各策略的p值比较
        ax1 = fig.add_subplot(gs[0, 0])
        if strategies:
            p_values = [get_shuffle_result(s, 'p_value_conservative', 1.0) for s in strategies]
            colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]
            bars = ax1.barh(strategies, p_values, color=colors)
            ax1.axvline(0.05, color='black', linestyle='--', lw=2, label='α=0.05')
            ax1.set_xlabel('p-value')
            ax1.set_title(f'Significance ({shuffle_mode})', fontweight='bold')
            ax1.legend()

        # 2. 效应量比较
        ax2 = fig.add_subplot(gs[0, 1])
        if strategies:
            effects = [get_shuffle_result(s, 'effect_size', 0) for s in strategies]
            max_effect = max(max(effects), 1) if effects else 1
            colors = plt.cm.RdYlGn(np.array(effects) / max_effect * 0.5 + 0.5)
            ax2.barh(strategies, effects, color=colors)
            ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(0.8, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Effect Size (Cohen\'s d)')
            ax2.set_title('Effect Size', fontweight='bold')

        # 3. 分组数量
        ax3 = fig.add_subplot(gs[0, 2])
        if strategies:
            n_groups = [get_shuffle_result(s, 'n_groups', 0) for s in strategies]
            ax3.barh(strategies, n_groups, color='steelblue')
            ax3.set_xlabel('Number of Groups')
            ax3.set_title('Groups per Strategy', fontweight='bold')

        # 4. 一致性分布比较（中间行）
        ax4 = fig.add_subplot(gs[1, :])
        group_stats_all = all_results.get('group_stats', {})
        if group_stats_all:
            data_for_box = []
            labels_for_box = []
            for strategy, stats in group_stats_all.items():
                cons = [s.direction_consistency for s in stats.values()]
                data_for_box.append(cons)
                labels_for_box.append(strategy)

            if data_for_box:
                bp = ax4.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
                colors = plt.cm.Set2(np.linspace(0, 1, len(data_for_box)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                ax4.set_ylabel('Direction Consistency')
                ax4.set_title('Consistency Distribution Across Strategies', fontweight='bold')
                ax4.axhline(0, color='black', lw=0.5)

        # 5. 汇总文本
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        summary_text = "═" * 60 + "\n"
        summary_text += f"        VALIDATION SUMMARY (shuffle={shuffle_mode})\n"
        summary_text += "═" * 60 + "\n\n"
        summary_text += f"{'Strategy':<22} {'p(unw)':<8} {'p(wei)':<8} {'eff':<6}\n"
        summary_text += "-" * 50 + "\n"

        for strategy in strategies:
            p = get_shuffle_result(strategy, 'p_value_conservative', 1.0)
            p_w = get_shuffle_result(strategy, 'p_value_weighted_conservative', 1.0)
            e = get_shuffle_result(strategy, 'effect_size', 0)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            summary_text += f"{strategy[:22]:<22} {p:<8.4f} {p_w:<8.4f} {e:<6.2f} {sig}\n"

        summary_text += "\n" + "═" * 60 + "\n"

        # 总结
        significant_strategies = [s for s in strategies
                                  if get_shuffle_result(s, 'p_value_conservative', 1.0) < 0.05]
        significant_weighted = [s for s in strategies
                                if get_shuffle_result(s, 'p_value_weighted_conservative', 1.0) < 0.05]

        if significant_strategies:
            summary_text += f"\n✓ Unweighted: {len(significant_strategies)}/{len(strategies)} significant\n"
            summary_text += f"✓ Weighted:   {len(significant_weighted)}/{len(strategies)} significant\n"
            summary_text += "  EC rules show geometric fingerprints in embedding space\n"
        else:
            summary_text += "\n⚠ No strategy shows significant patterns (unweighted)\n"

        ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
                 fontsize=11, family='monospace', ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.suptitle(f'EC Rule Validation Dashboard ({shuffle_mode})', fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(self.output_dir / f'validation_dashboard_{shuffle_mode}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: validation_dashboard_{shuffle_mode}.png")


# ============================================================================
# PART 8: Main Pipeline
# ============================================================================

def run_validation(
        cache_dir: str = '../kegg_cache',
        tgsc_file: str = '../02_kegg_mapping/tgsc_to_kegg.csv',
        model_dir: str = '../experiments',
        output_dir: str = './ec_validation_results',
        ec_level: int = 3,
        n_permutations: int = 1000,
        strategies: List[str] = None,
        min_group_size: int = 5,
        embedding_layer_name: str = None,
        use_prediction_space: bool = False,
        dedup_mode: str = 'src_tgt_ec',
        shuffle_modes: List[str] = None,
        max_samples: int = 20000
):
    """
    运行完整的验证流程

    Args:
        cache_dir: 路径缓存目录
        tgsc_file: TGSC-KEGG映射文件
        model_dir: OpenPOM模型目录
        output_dir: 输出目录
        ec_level: EC截取层级 (1/2/3)
        n_permutations: 置换检验次数
        strategies: 要测试的策略列表（默认全部）
        min_group_size: 最小分组大小
        embedding_layer_name: OpenPOM中间层名称（None则自动检测）
        use_prediction_space: 是否使用预测输出空间（不推荐）
        dedup_mode: 去重模式 ('none', 'src_tgt_ec', 'src_tgt')
        shuffle_modes: 置换模式列表 (默认 ['global', 'within_odor_pair'])
        max_samples: shuffle test最大样本数（超过则采样）
    """
    print("=" * 80)
    print("EC Rule Validation Pipeline (V2)")
    print(f"EC Level: {ec_level}")
    print(f"Min Group Size: {min_group_size}")
    print(f"Dedup Mode: {dedup_mode}")
    if use_prediction_space:
        print("⚠ Using PREDICTION SPACE (not latent embedding)")
    else:
        print("Using LATENT EMBEDDING space")
    print("=" * 80)

    # 默认测试所有策略
    if strategies is None:
        strategies = DEFAULT_STRATEGIES

    # 默认置换模式
    if shuffle_modes is None:
        shuffle_modes = ['global', 'within_odor_pair']

    # Step 1: 加载数据
    loader = PathwayDataLoader(cache_dir=cache_dir, tgsc_file=tgsc_file)
    if not loader.load_all():
        print("\n❌ Failed to load data")
        return None

    if not loader.edges:
        print("\n❌ No valid edges found")
        return None

    # Step 1.5: 去重（关键！）
    original_edge_count = len(loader.edges)
    if dedup_mode != 'none':
        loader.deduplicate_edges(mode=dedup_mode)

    dedup_ratio = 1 - len(loader.edges) / original_edge_count if original_edge_count > 0 else 0

    # Step 2: 提取嵌入
    print("\n" + "=" * 80)
    print("Extracting Embeddings")
    print("=" * 80)

    extractor = EmbeddingExtractor(
        model_dir=model_dir,
        embedding_layer_name=embedding_layer_name,
        use_prediction_space=use_prediction_space
    )
    extractor.initialize()

    # 收集所有SMILES
    all_smiles = set()
    for edge in loader.edges:
        if edge.source_smiles:
            all_smiles.add(edge.source_smiles)
        if edge.target_smiles:
            all_smiles.add(edge.target_smiles)

    embeddings = extractor.get_batch_embeddings(list(all_smiles))
    print(f"  ✓ Got embeddings for {len(embeddings)} compounds")

    # 清理hook
    extractor.cleanup()

    # Step 3: 初始化分析器
    analyzer = EndToEndDeltaAnalyzer(loader.edges, embeddings, ec_level=ec_level)
    analyzer.compute_deltas()

    # Step 4: 运行各策略分析
    print("\n" + "=" * 80)
    print("Running Analyses")
    print("=" * 80)

    all_results = {
        'shuffle_tests': {},  # 改为嵌套结构: {strategy: {shuffle_mode: results}}
        'separations': {},
        'group_stats': {},
        'filtering_stats': {},
        'dedup_stats': {
            'original_edges': original_edge_count,
            'after_dedup': len(loader.edges),
            'dedup_ratio': dedup_ratio,
            'dedup_mode': dedup_mode
        }
    }

    for strategy in strategies:
        print(f"\n{'─' * 40}")
        print(f"Strategy: {strategy}")
        print(f"{'─' * 40}")

        # 分组分析
        stats = analyzer.analyze_by_strategy(strategy, min_group_size=min_group_size)
        all_results['group_stats'][strategy] = stats

        # 记录过滤统计（使用 unique edge count）
        total_deltas = len(analyzer.deltas)

        # 统计 unique edges（避免 multi-label 展开导致的重复计数）
        unique_edge_ids = set()
        for stat in stats.values():
            unique_edge_ids.update(stat.edge_ids)
        edges_in_groups = len(unique_edge_ids)

        # membership 计数（用于参考）
        membership_count = sum(s.n_edges for s in stats.values())

        all_results['filtering_stats'][strategy] = {
            'total_edges': total_deltas,
            'unique_edges_in_valid_groups': edges_in_groups,
            'membership_count': membership_count,  # multi-label 展开后的计数
            'edges_filtered_out': total_deltas - edges_in_groups,
            'filter_ratio': 1 - edges_in_groups / total_deltas if total_deltas > 0 else 0,
            'n_groups_total': len(stats),
            'is_multilabel_expanded': membership_count > edges_in_groups,
        }

        if membership_count > edges_in_groups:
            print(
                f"    Filtering: {edges_in_groups}/{total_deltas} unique edges ({(1 - all_results['filtering_stats'][strategy]['filter_ratio']) * 100:.1f}%)")
            print(
                f"    ⚠ Multi-label expansion: {membership_count} group-memberships (avg {membership_count / edges_in_groups:.1f}× per edge)")
        else:
            print(
                f"    Filtering: {edges_in_groups}/{total_deltas} edges retained ({(1 - all_results['filtering_stats'][strategy]['filter_ratio']) * 100:.1f}%)")

        if stats:
            # 对每种shuffle_mode运行置换检验
            all_results['shuffle_tests'][strategy] = {}
            for shuffle_mode in shuffle_modes:
                shuffle_result = analyzer.run_shuffle_test(
                    strategy,
                    n_permutations=n_permutations,
                    shuffle_mode=shuffle_mode,
                    max_samples=max_samples
                )
                all_results['shuffle_tests'][strategy][shuffle_mode] = shuffle_result

            # 组间分离度
            separation = analyzer.compute_between_group_separation(strategy)
            all_results['separations'][strategy] = separation

    # Step 5: 可视化
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)

    visualizer = ValidationVisualizer(output_dir=output_dir)

    for strategy in strategies:
        # 使用global模式的结果绘制单策略图（如果存在）
        if strategy in all_results['shuffle_tests']:
            shuffle_tests = all_results['shuffle_tests'][strategy]
            if 'global' in shuffle_tests:
                visualizer.plot_shuffle_test(shuffle_tests['global'], f"{strategy}_global")

        if strategy in all_results['group_stats']:
            visualizer.plot_group_consistency_distribution(
                all_results['group_stats'][strategy], strategy)
            visualizer.plot_top_groups(all_results['group_stats'][strategy], strategy)
            visualizer.plot_pca_with_arrows(analyzer, strategy)

    # 为每种shuffle_mode生成dashboard
    for shuffle_mode in shuffle_modes:
        visualizer.plot_summary_dashboard(all_results, shuffle_mode=shuffle_mode)

    # Step 6: 导出结果
    print("\n" + "=" * 80)
    print("Exporting Results")
    print("=" * 80)

    export_data = {
        'parameters': {
            'ec_level': ec_level,
            'n_permutations': n_permutations,
            'strategies': strategies,
            'min_group_size': min_group_size,
            'embedding_layer': extractor.embedding_layer_name,
            'use_prediction_space': use_prediction_space,
            'dedup_mode': dedup_mode,
            'shuffle_modes': shuffle_modes
        },
        'summary': {
            'n_edges_original': original_edge_count,
            'n_edges_after_dedup': len(loader.edges),
            'dedup_ratio': dedup_ratio,
            'n_compounds': len(embeddings),
            'embedding_dim': extractor.embedding_dim,
            'embedding_type': 'prediction_space' if use_prediction_space else 'latent_embedding'
        },
        'filtering_stats': all_results['filtering_stats'],
        'results_by_strategy': {}
    }

    for strategy in strategies:
        export_data['results_by_strategy'][strategy] = {
            'by_shuffle_mode': {}
        }

        shuffle_tests = all_results['shuffle_tests'].get(strategy, {})
        for shuffle_mode in shuffle_modes:
            shuffle_res = shuffle_tests.get(shuffle_mode, {})
            sep_res = all_results['separations'].get(strategy, {})
            filter_res = all_results['filtering_stats'].get(strategy, {})

            export_data['results_by_strategy'][strategy]['by_shuffle_mode'][shuffle_mode] = {
                'n_groups': shuffle_res.get('n_groups', 0),
                'n_samples': shuffle_res.get('n_samples', 0),
                # 非加权结果
                'p_value': shuffle_res.get('p_value_conservative', None),
                'effect_size': shuffle_res.get('effect_size', None),
                'real_consistency': shuffle_res.get('real_mean_consistency', None),
                'null_consistency_mean': shuffle_res.get('shuffled_mean', None),
                'null_consistency_std': shuffle_res.get('shuffled_std', None),
                # 加权结果
                'p_value_weighted': shuffle_res.get('p_value_weighted_conservative', None),
                'effect_size_weighted': shuffle_res.get('effect_size_weighted', None),
                'real_consistency_weighted': shuffle_res.get('real_mean_weighted', None),
            }

        # 分离度（不依赖shuffle_mode）
        sep_res = all_results['separations'].get(strategy, {})
        export_data['results_by_strategy'][strategy]['separation'] = {
            'avg_cosine_separation': sep_res.get('avg_cosine_separation', None),
            'avg_euclidean_separation': sep_res.get('avg_euclidean_separation', None),
        }
        export_data['results_by_strategy'][strategy]['filter_ratio'] = \
            all_results['filtering_stats'].get(strategy, {}).get('filter_ratio', None)

    results_file = Path(output_dir) / 'validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    print(f"  ✓ Results saved to: {results_file}")

    # 最终总结
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    # p-value 分辨率警告
    min_p = 1 / (n_permutations + 1)
    if n_permutations < 500:
        print(f"\n   ⚠ WARNING: n_permutations={n_permutations} is low")
        print(f"     Minimum detectable p-value = {min_p:.4f}")
        print(f"     Consider using --n-permutations 1000 for publication-quality results")

    embedding_type = "PREDICTION SPACE" if use_prediction_space else "LATENT EMBEDDING"
    print(f"\n📊 Results Summary")
    print(f"   EC Level: {ec_level}, Embedding: {embedding_type}")
    print(f"   Dedup: {dedup_mode} ({original_edge_count} → {len(loader.edges)} edges, -{dedup_ratio * 100:.1f}%)")
    print(f"   {len(embeddings)} compounds with embeddings")
    if extractor.embedding_layer_name:
        print(f"   Embedding layer: {extractor.embedding_layer_name}")

    # 按shuffle_mode显示结果
    for shuffle_mode in shuffle_modes:
        print(f"\n   ═══ Shuffle Mode: {shuffle_mode} ═══")
        print(f"   {'Strategy':<25} {'p(unw)':<10} {'p(wei)':<10} {'effect':<10} {'Sig?'}")
        print(f"   {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 5}")

        significant_count = 0
        significant_weighted_count = 0

        for strategy in strategies:
            shuffle_tests = all_results['shuffle_tests'].get(strategy, {})
            res = shuffle_tests.get(shuffle_mode, {})
            p = res.get('p_value_conservative', 1.0)
            p_w = res.get('p_value_weighted_conservative', 1.0)
            e = res.get('effect_size', 0)
            sig = "✓" if p < 0.05 else "✗"
            sig_w = "✓" if p_w < 0.05 else "✗"
            print(f"   {strategy:<25} {p:<10.4f} {p_w:<10.4f} {e:<10.2f} {sig}/{sig_w}")
            if p < 0.05:
                significant_count += 1
            if p_w < 0.05:
                significant_weighted_count += 1

        print(
            f"   → Significant: {significant_count}/{len(strategies)} (unw), {significant_weighted_count}/{len(strategies)} (wei)")

    # 关键对比：global vs within_odor_pair
    if 'global' in shuffle_modes and 'within_odor_pair' in shuffle_modes:
        print(f"\n   ═══ Key Comparison: Global vs Within-Odor-Pair ═══")
        print(f"   {'Strategy':<20} {'Global p':<10} {'Cond. p':<10} {'Δ(real-null)':<12} {'Conclusion'}")
        print(f"   {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 25}")

        # 定义哪些策略的 within_odor_pair 检验是无效的
        INVALID_COMBINATIONS = {
            'odor_pair': 'within_odor_pair',  # 分组键 == 约束键
            'full_combination': 'within_odor_pair',  # 包含 odor_pair 作为子键
        }

        for strategy in strategies:
            shuffle_tests = all_results['shuffle_tests'].get(strategy, {})
            res_global = shuffle_tests.get('global', {})
            res_odor = shuffle_tests.get('within_odor_pair', {})

            p_global = res_global.get('p_value_conservative', 1.0)
            p_odor = res_odor.get('p_value_conservative', 1.0)
            delta = res_odor.get('real_null_delta', 0)  # real - null 差值
            invalid = res_odor.get('invalid_test', False) or strategy in INVALID_COMBINATIONS

            # 更严谨的判定逻辑
            if invalid:
                conclusion = "⊘ Invalid test (skip)"
            elif p_global >= 0.05:
                conclusion = "✗ Not significant"
            elif p_odor >= 0.05:
                conclusion = "⚠ Odor-driven (confound)"
            elif delta < 0.01:  # real-null 差距太小
                conclusion = "△ Marginal (Δ<0.01)"
            else:
                conclusion = "✓ EC fingerprint"

            delta_str = f"{delta:.4f}" if not invalid else "N/A"
            print(f"   {strategy:<20} {p_global:<10.4f} {p_odor:<10.4f} {delta_str:<12} {conclusion}")

    print(f"\n📁 All results saved to: {output_dir}/")

    return all_results


# ============================================================================
# PART 9: Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='EC Rule Validation in OpenPOM Space')
    parser.add_argument('--cache-dir', default='../kegg_cache', help='Pathway cache directory')
    parser.add_argument('--tgsc-file', default='../02_kegg_mapping/tgsc_to_kegg.csv', help='TGSC-KEGG mapping file')
    parser.add_argument('--model-dir', default='../experiments', help='OpenPOM model directory')
    parser.add_argument('--output-dir', default='./ec_validation_results', help='Output directory')
    parser.add_argument('--ec-level', type=int, default=3, choices=[1, 2, 3],
                        help='EC truncation level (1/2/3)')
    parser.add_argument('--n-permutations', type=int, default=1000, help='Number of permutations')
    parser.add_argument('--strategies', nargs='+', default=None,
                        help='Strategies to test (default: EC-only subset)')
    parser.add_argument('--min-group-size', type=int, default=5,
                        help='Minimum edges per group (default: 5)')
    parser.add_argument('--embedding-layer', type=str, default=None,
                        help='Name of embedding layer to hook (auto-detect if not specified)')
    parser.add_argument('--use-prediction-space', action='store_true',
                        help='Use prediction output instead of latent embedding (not recommended)')
    parser.add_argument('--dedup-mode', type=str, default='src_tgt_ec',
                        choices=['none', 'src_tgt_ec', 'src_tgt'],
                        help='Deduplication mode (default: src_tgt_ec)')
    parser.add_argument('--shuffle-modes', nargs='+',
                        default=['global', 'within_odor_pair'],
                        choices=['global', 'within_source', 'within_odor_pair', 'within_length'],
                        help='Shuffle modes for permutation test (default: global within_odor_pair)')
    parser.add_argument('--max-samples', type=int, default=20000,
                        help='Max samples for shuffle test (default: 20000, 0=no limit)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling/visualization (default: 42)')

    args = parser.parse_args()

    import random

    max_samples = args.max_samples if args.max_samples > 0 else float('inf')
    np.random.seed(args.seed)
    random.seed(args.seed)

    results = run_validation(
        cache_dir=args.cache_dir,
        tgsc_file=args.tgsc_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        ec_level=args.ec_level,
        n_permutations=args.n_permutations,
        strategies=args.strategies,
        min_group_size=args.min_group_size,
        embedding_layer_name=args.embedding_layer,
        use_prediction_space=args.use_prediction_space,
        dedup_mode=args.dedup_mode,
        shuffle_modes=args.shuffle_modes,
        max_samples=max_samples
    )
