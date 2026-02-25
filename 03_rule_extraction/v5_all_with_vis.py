#!/usr/bin/env python3
"""
气味规则提取与可视化系统 - 整合版
"""

import pandas as pd
import numpy as np
import networkx as nx
import requests
import json
import time
import os
import random
import pickle
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from itertools import combinations

warnings.filterwarnings('ignore')
np.random.seed(42)

# Matplotlib 配置
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

import pandas as pd
import numpy as np
import networkx as nx
import requests
import json
import time
import os
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import pickle
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from itertools import combinations

warnings.filterwarnings('ignore')
np.random.seed(42)

# 背景EC：太常见、缺乏区分度的EC类
BACKGROUND_EC = {'1.1.1'}  # 醇脱氢酶，几乎每条路径都有

# ✅ 新增：载体分子黑名单（高频中间节点，会污染路径）
UBIQUITOUS_COMPOUNDS = {
    # 水 / 质子 / 简单无机物
    "C00001",  # H2O (水)
    "C00007",  # O2 (氧气)
    "C00080",  # H+ (质子)
    "C00011",  # CO2 (二氧化碳)
    "C00014",  # NH3 (氨)
    "C00059",  # SO4 (硫酸根)
    "C00283",  # H2S (硫化氢)

    # 磷酸 / 能量载体
    "C00002",  # ATP
    "C00008",  # ADP
    "C00020",  # AMP
    "C00013",  # PPi (焦磷酸)
    "C00009",  # Pi (正磷酸)
    "C00044",  # GTP
    "C00035",  # GDP
    "C00075",  # UTP
    "C00015",  # UDP
    "C00063",  # CTP
    "C00112",  # CDP

    # 氧化还原辅酶
    "C00003",  # NAD+
    "C00004",  # NADH
    "C00005",  # NADPH
    "C00006",  # NADP+
    "C00016",  # FAD
    "C01352",  # FADH2
    "C00061",  # FMN
    "C01847",  # FMNH2

    # 辅酶A相关
    "C00010",  # CoA
    "C00024",  # Acetyl-CoA
    "C00083",  # Malonyl-CoA
    "C00091",  # Succinyl-CoA

    # 其他常见载体
    "C00019",  # S-Adenosyl-L-methionine (SAM)
    "C00021",  # S-Adenosyl-L-homocysteine (SAH)
    "C00101",  # THF (四氢叶酸)
    "C00234",  # 10-Formyl-THF
    "C00440",  # 5-Methyl-THF
    "C00053",  # 3'-Phosphoadenosine 5'-phosphosulfate (PAPS)
    "C00054",  # Adenosine 3',5'-bisphosphate
}


# =============================================================================
# PART 1: 数据生成部分 - 从 paste1 复制
# =============================================================================
# 复制 paste1 的第 35-620 行，包括：
#   - to_ec_level() 函数 (第35-62行)
#   - OdorPathwayAnalyzer 类全部 (第65-620行左右)
# =============================================================================
def to_ec_level(ec: str, level: int = 3) -> str:
    """
    截取 EC 编号到指定级别

    Args:
        ec: 原始 EC 编号，如 '1.14.14.130'
        level: 目标级别，2/3/4，默认 3

    Returns:
        截取后的 EC，如 '1.14.14'；无效则返回空字符串

    Examples:
        > > > to_ec_level('1.14.14.130', 3)
        '1.14.14'
        > > > to_ec_level('4.2.1.-', 3)
        '4.2.1'
        > >> to_ec_level('1.2.-.-', 2)
        '1.2'
        > >> to_ec_level('1.2.-.-', 3)
        ''
    """
    if not ec or '.' not in ec:
        return ''

    parts = ec.split('.')

    # 检查是否有足够的级别
    if len(parts) < level:
        return ''

    # 截取到目标级别
    result_parts = parts[:level]

    # 过滤掉最后一位是通配符的情况
    if result_parts[-1] == '-':
        return ''

    return '.'.join(result_parts)

class OdorPathwayAnalyzer:
    """Optimized analyzer with pathway caching for debugging"""

    @staticmethod
    def parse_kegg_flat(text: str) -> dict:
        """解析 KEGG 平面格式文本，正确处理续行

        KEGG 格式规则：
        - 字段名占前 12 个字符
        - 续行时前 12 个字符为空格
        """
        fields = {}
        current_key = None

        for raw_line in text.splitlines():
            key = raw_line[:12].strip()
            val = raw_line[12:].strip()

            if key:  # 新字段开始
                current_key = key
                if current_key not in fields:
                    fields[current_key] = []
                if val:
                    fields[current_key].append(val)
            elif current_key and val:  # 续行
                fields[current_key].append(val)

        # 合并每个字段的所有行为单个字符串
        return {k: " ".join(v) for k, v in fields.items()}

    def __init__(self, cache_dir="../kegg_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Network components
        self.compound_to_reactions = defaultdict(set)
        self.reaction_to_compounds = defaultdict(set)
        self.reaction_reversibility = {}
        self.ec_data = {}

        # ✅ 新增：Main-pair 数据结构
        self.main_pairs_raw = {}  # R → [(C1, C2), ...]
        self.main_pairs_index = {}  # R → {C_in: {C_out1, C_out2, ...}}

        # TGSC data
        self.tgsc_data = None
        self.odor_attributes = []
        self.compound_profiles = {}
        self.odorous_compounds = set()  # Only compounds with odor

        # Processed data
        self.reaction_graph = None
        self.pathways = []

    def load_tgsc_data(self, csv_file="../02_kegg_mapping/tgsc_to_kegg.csv"):
        """Load TGSC data and identify odorous compounds only"""
        print(f"Loading TGSC-KEGG data from {csv_file}...")

        try:
            self.tgsc_data = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(self.tgsc_data)} TGSC compounds")

            # Identify odor attributes
            core_cols = {'TGSC ID', 'KEGG_IDs', 'CID', 'IsomericSMILES', 'IUPACName',
                         'Updated_Desc_v2', 'Solvent'}

            self.odor_attributes = []
            for col in self.tgsc_data.columns:
                if col not in core_cols:
                    unique_vals = self.tgsc_data[col].dropna().unique()
                    if len(unique_vals) <= 2 and all(v in [0, 1, 0.0, 1.0] for v in unique_vals):
                        self.odor_attributes.append(col)

            print(f"✓ Identified {len(self.odor_attributes)} odor attributes")

            # Filter compounds with valid KEGG IDs
            valid_kegg = self.tgsc_data['KEGG_IDs'].notna() & self.tgsc_data['KEGG_IDs'].str.match(r'^C\d{5}$')
            valid_data = self.tgsc_data[valid_kegg].copy()

            # Create profiles and identify ODOROUS compounds only
            odorless_count = 0
            odorous_count = 0

            for _, row in valid_data.iterrows():
                kegg_id = row['KEGG_IDs']
                odor_profile = row[self.odor_attributes].fillna(0).values.astype(np.int8)

                # Check if compound has any odor (excluding 'odorless' attribute)
                non_odorless_attributes = [attr for attr in self.odor_attributes if 'odorless' not in attr.lower()]
                odor_indices = [self.odor_attributes.index(attr) for attr in non_odorless_attributes]
                has_odor = any(odor_profile[i] == 1 for i in odor_indices)

                if has_odor:
                    # Only keep compounds WITH odor
                    self.odorous_compounds.add(kegg_id)
                    odorous_count += 1

                    self.compound_profiles[kegg_id] = {
                        'tgsc_id': str(row['TGSC ID']),
                        'name': str(row['IUPACName']) if pd.notna(row['IUPACName']) else f"Compound_{kegg_id}",
                        'descriptors': str(row.get('Updated_Desc_v2', '')),
                        'odor_profile': odor_profile,
                        'odor_count': np.sum(odor_profile),
                        'dominant_odors': [self.odor_attributes[i] for i, v in enumerate(odor_profile) if v == 1][:3]
                    }
                else:
                    odorless_count += 1

            print(f"✓ Identified {odorous_count} ODOROUS compounds (excluded {odorless_count} odorless)")
            print(f"✓ Working with {len(self.odorous_compounds)} compounds for pathway analysis")

            return True

        except Exception as e:
            print(f"❌ Error loading TGSC data: {e}")
            return False

    def download_kegg_data(self):
        """Download KEGG reaction data"""
        print("Downloading KEGG reaction network...")

        # Download compound -> reaction links
        comp_react_file = self.cache_dir / "compound_reaction_links.tsv"
        if not comp_react_file.exists():
            print("  Fetching compound->reaction links...")
            url = "https://rest.kegg.jp/link/reaction/compound"
            response = requests.get(url)
            if response.status_code == 200:
                with open(comp_react_file, 'w') as f:
                    f.write(response.text)

        # Download reaction -> compound links
        react_comp_file = self.cache_dir / "reaction_compound_links.tsv"
        if not react_comp_file.exists():
            print("  Fetching reaction->compound links...")
            url = "https://rest.kegg.jp/link/compound/reaction"
            response = requests.get(url)
            if response.status_code == 200:
                with open(react_comp_file, 'w') as f:
                    f.write(response.text)

        # Parse links
        self._parse_kegg_links(comp_react_file, react_comp_file)

    def _parse_kegg_links(self, comp_react_file, react_comp_file):
        """Parse KEGG links efficiently"""
        # Parse compound -> reaction ONLY for odorous compounds
        with open(comp_react_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    compound = parts[0].replace('cpd:', '')
                    reaction = parts[1].replace('rn:', '')

                    # ✅ 过滤载体分子
                    if compound in UBIQUITOUS_COMPOUNDS:
                        continue

                    # Only store if compound is odorous
                    if compound in self.odorous_compounds:
                        self.compound_to_reactions[compound].add(reaction)

        # Parse reaction -> compound
        with open(react_comp_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    reaction = parts[0].replace('rn:', '')
                    compound = parts[1].replace('cpd:', '')

                    # ✅ 过滤载体分子
                    if compound in UBIQUITOUS_COMPOUNDS:
                        continue

                    self.reaction_to_compounds[reaction].add(compound)



        print(
            f"✓ Parsed {len(self.compound_to_reactions)} odorous compounds and {len(self.reaction_to_compounds)} reactions")
        print(f"  (已过滤 {len(UBIQUITOUS_COMPOUNDS)} 种载体分子)")

    def download_expanded_reaction_metadata(self, expanded_reactions):
        """Download metadata with robust caching (using working v17 approach)"""
        print("Downloading metadata with robust caching...")
        print(f"  Requested reactions: {len(expanded_reactions)}")

        cache_file = self.cache_dir / "complete_reaction_cache.json"

        # Load existing cache using the working v17 approach
        cache_data = {
            'successful': {
                'ec_data': {},
                'reversibility': {}
            },
            'failed': {
                'reactions': set(),
                'last_attempt': {},
                'attempt_count': {}
            },
            'metadata': {
                'last_updated': None,
                'total_attempts': 0,
                'success_rate': 0.0
            }
        }

        # Enhanced cache loading
        if cache_file.exists():
            print(f"  Loading cache: {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    enhanced_data = json.load(f)

                    # Handle different cache formats
                    if 'successful' in enhanced_data:
                        # New format with successful/failed structure
                        enhanced_data['failed']['reactions'] = set(enhanced_data['failed']['reactions'])
                        cache_data['successful']['ec_data'].update(
                            enhanced_data.get('successful', {}).get('ec_data', {}))
                        cache_data['successful']['reversibility'].update(
                            enhanced_data.get('successful', {}).get('reversibility', {}))
                        cache_data['failed'] = enhanced_data.get('failed', cache_data['failed'])
                        cache_data['metadata'] = enhanced_data.get('metadata', cache_data['metadata'])
                    else:
                        # Old format - direct ec_data mapping
                        for reaction_id, ec_numbers in enhanced_data.items():
                            if isinstance(ec_numbers, list):
                                cache_data['successful']['ec_data'][reaction_id] = ec_numbers
                                cache_data['successful']['reversibility'][reaction_id] = True  # Assume reversible

                    successful_count = len(cache_data['successful']['ec_data'])
                    failed_count = len(cache_data['failed']['reactions'])
                    print(f"    ✅ Cache loaded: {successful_count} successful, {failed_count} failed")
            except Exception as e:
                print(f"    ❌ Error loading cache: {e}")

        # Update instance variables
        self.ec_data = cache_data['successful']['ec_data'].copy()
        self.reaction_reversibility = cache_data['successful']['reversibility'].copy()

        # Determine what needs downloading
        cached_successful = set(cache_data['successful']['ec_data'].keys())
        previously_failed = cache_data['failed']['reactions']

        # never_attempted = expanded_reactions - cached_successful - previously_failed
        # need_retry = previously_failed & expanded_reactions

        # # ✅ 把 EC 为空的"成功"缓存也纳入重试
        # empty_ec_reactions = {
        #     rid for rid, ecs in cache_data['successful']['ec_data'].items()
        #     if not ecs and rid in expanded_reactions
        # }
        # never_attempted = expanded_reactions - cached_successful - previously_failed
        # need_retry = (previously_failed | empty_ec_reactions) & expanded_reactions
        # print(f"    🔄 Empty EC retries: {len(empty_ec_reactions & expanded_reactions)}")

        # ✅ 不重试空EC反应 - 空EC是正常情况，不是错误
        never_attempted = expanded_reactions - cached_successful - previously_failed
        need_retry = previously_failed & expanded_reactions  # 只重试真正失败的（HTTP错误等）




        print(f"  📊 Download analysis:")
        print(f"    ✅ Already successful: {len(cached_successful & expanded_reactions)}")
        print(f"    🆕 Never attempted: {len(never_attempted)}")
        print(f"    🔄 Need retry: {len(need_retry)}")

        if not never_attempted and not need_retry:
            print(f"  🎉 All requested reactions already cached successfully!")
            return

        # Download missing reactions
        to_download = never_attempted.copy()
        max_retry_attempts = 3

        for reaction in need_retry:
            attempt_count = cache_data['failed']['attempt_count'].get(reaction, 0)
            if attempt_count < max_retry_attempts:
                to_download.add(reaction)

        if not to_download:
            print(f"  ✅ No new downloads needed")
            return

        print(f"  📥 Downloading {len(to_download)} reactions...")

        # Download with tracking
        new_successful = {}
        new_reversibility = {}
        new_failures = set()
        download_stats = {'attempts': 0, 'successes': 0, 'failures': 0}

        for reaction_id in tqdm(to_download, desc="Downloading reactions"):
            download_stats['attempts'] += 1

            try:
                url = f"https://rest.kegg.jp/get/{reaction_id}"
                response = requests.get(url, timeout=15)

                if response.status_code == 200:
                    text = response.text

                    # if 'ENTRY' in text and reaction_id in text:
                    #     # Extract EC numbers and reversibility
                    #     ec_numbers = []
                    #     is_reversible = True
                    #
                    #     for line in text.split('\n'):
                    #         if line.startswith('ENZYME'):
                    #             ec_numbers.extend(line.split()[1:])
                    #         elif line.startswith('EQUATION'):
                    #             equation = line[8:].strip()
                    #             is_reversible = ('⇄' in equation or '<=>' in equation or
                    #                              '<==>' in equation or '<->' in equation)
                    #
                    #     new_successful[reaction_id] = ec_numbers
                    #     new_reversibility[reaction_id] = is_reversible

                    # if 'ENTRY' in text and reaction_id in text:
                    #     # ✅ 使用新的解析器处理续行
                    #     fields = self.parse_kegg_flat(text)
                    #
                    #     # 1) 提取 EC 编号（支持通配符 -）
                    #     ec_numbers = []
                    #     enzyme_str = fields.get("ENZYME", "")
                    #     if enzyme_str:
                    #         # 匹配标准 EC：1.2.3.4 或带通配符：1.2.3.- / 1.2.-.-
                    #         import re
                    #         ec_numbers = re.findall(r'\d+(?:\.\d+|\.-){1,3}', enzyme_str)

                    if 'ENTRY' in text and reaction_id in text:
                        # ✅ 使用新的解析器处理续行
                        fields = self.parse_kegg_flat(text)

                        import re
                        ec_numbers = []

                        # ✅ 策略 1: 优先从 ENZYME 字段提取（直接列出的 EC）
                        enzyme_str = fields.get("ENZYME", "")
                        if enzyme_str:
                            # 匹配：1.2.3.4 或 1.2.3.- 或 1.2.-.-
                            ec_numbers = re.findall(r'\d+(?:\.\d+|\.-){1,3}', enzyme_str)

                        # # ✅ 策略 2: 如果 ENZYME 为空，从 ORTHOLOGY 字段提取
                        # if not ec_numbers:
                        #     orthology_str = fields.get("ORTHOLOGY", "")
                        #     if orthology_str:
                        #         # ORTHOLOGY 格式: "K16873 ... [EC:1.1.3.47 1.1.3.-]"
                        #         # 提取方括号内的 EC 编号
                        #         ec_matches = re.findall(r'\[EC:([\d\.\-\s]+)\]', orthology_str)
                        #         for match in ec_matches:
                        #             # match 可能是 "1.1.3.47 1.1.3.-"，需要分割提取
                        #             ecs_in_bracket = re.findall(r'\d+(?:\.\d+|\.-){1,3}', match)
                        #             ec_numbers.extend(ecs_in_bracket)
                        # ✅ 策略 2: 从 ORTHOLOGY 字段提取
                        if not ec_numbers:
                            orthology_str = fields.get("ORTHOLOGY", "")
                            if orthology_str:
                                ec_matches = re.findall(r'\[EC:([\d\.\-\s]+)\]', orthology_str)
                                for match in ec_matches:
                                    # ✅ 提取三级或四级的EC（包括通配符）
                                    ecs = re.findall(r'\d+(?:\.\d+|\.-){1,3}', match)
                                    ec_numbers.extend(ecs)

                        # ✅ 去重保持顺序
                        if ec_numbers:
                            ec_numbers = list(dict.fromkeys(ec_numbers))

                        print(ec_numbers)

                        # 2) 判断可逆性（合并后的完整 EQUATION）
                        equation_str = fields.get("EQUATION", "")
                        is_reversible = any(sym in equation_str for sym in
                                            ["⇄", "<=>", "<==>", "<->"])

                        new_successful[reaction_id] = ec_numbers
                        new_reversibility[reaction_id] = is_reversible


                        download_stats['successes'] += 1
                        cache_data['failed']['reactions'].discard(reaction_id)
                    else:
                        new_failures.add(reaction_id)
                        download_stats['failures'] += 1
                else:
                    new_failures.add(reaction_id)
                    download_stats['failures'] += 1

                time.sleep(0.35)  # Rate limiting

            except Exception as e:
                new_failures.add(reaction_id)
                download_stats['failures'] += 1
                time.sleep(0.5)

        # Update cache
        cache_data['successful']['ec_data'].update(new_successful)
        cache_data['successful']['reversibility'].update(new_reversibility)
        cache_data['failed']['reactions'].update(new_failures)

        current_time = time.time()
        for reaction in new_failures:
            cache_data['failed']['last_attempt'][reaction] = current_time
            cache_data['failed']['attempt_count'][reaction] = cache_data['failed']['attempt_count'].get(reaction, 0) + 1

        # Update instance variables
        self.ec_data.update(new_successful)
        self.reaction_reversibility.update(new_reversibility)

        # Save cache
        self._save_enhanced_cache(cache_data, cache_file)

        success_rate = download_stats['successes'] / download_stats['attempts'] if download_stats['attempts'] > 0 else 0
        print(
            f"  📊 Download Results: {download_stats['successes']} successful, {download_stats['failures']} failed ({success_rate:.1%})")

    def _save_enhanced_cache(self, cache_data, cache_file):
        """Save enhanced cache"""
        try:
            cache_data_to_save = cache_data.copy()
            cache_data_to_save['failed']['reactions'] = list(cache_data['failed']['reactions'])

            with open(cache_file, 'w') as f:
                json.dump(cache_data_to_save, f, indent=2)

            file_size = cache_file.stat().st_size
            total_successful = len(cache_data['successful']['ec_data'])
            print(f"  💾 Cache saved: {total_successful} reactions, {file_size:,} bytes")

        except Exception as e:
            print(f"  💥 Error saving cache: {e}")

    def load_main_pairs(self, main_pairs_file: str = None):
        """加载 RCLASS main-pair 数据并构建快速索引

        Args:
            main_pairs_file: main_pairs.json 文件路径

        Returns:
            bool: 是否成功加载
        """
        if main_pairs_file is None:
            main_pairs_file = self.cache_dir / "rclass_verification" / "main_pairs.json"

        main_pairs_path = Path(main_pairs_file)

        if not main_pairs_path.exists():
            print(f"⚠️  Main-pair 文件不存在: {main_pairs_path}")
            print("   将不使用 main-pair 过滤（构图会更稀疏）")
            return False

        print(f"📦 加载 main-pair 数据...")
        try:
            with open(main_pairs_path, 'r') as f:
                self.main_pairs_raw = json.load(f)
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False

        # 构建快速索引：R → {C_in: {C_out}}
        from collections import defaultdict
        self.main_pairs_index = defaultdict(lambda: defaultdict(set))

        filtered_pairs_count = 0  # ✅ 统计过滤数量

        for reaction, pairs in self.main_pairs_raw.items():
            for pair in pairs:
                # JSON 格式是 [[c1, c2], ...] 而非 [(c1, c2), ...]
                if isinstance(pair, list) and len(pair) == 2:
                    c1, c2 = pair
                else:
                    continue

                # ✅ 过滤载体分子
                if c1 in UBIQUITOUS_COMPOUNDS or c2 in UBIQUITOUS_COMPOUNDS:
                    filtered_pairs_count += 1
                    continue

                # 前向：c1 → c2
                self.main_pairs_index[reaction][c1].add(c2)

                # 反向：只有可逆反应才添加 c2 → c1
                if self.reaction_reversibility.get(reaction, True):
                    self.main_pairs_index[reaction][c2].add(c1)

        # 统计
        indexed_reactions = len(self.main_pairs_index)
        total_pairs = sum(len(pairs) for pairs in self.main_pairs_raw.values())

        print(f"✅ 成功加载 {indexed_reactions} 个反应的 main-pair")
        print(f"   总计 {total_pairs} 个化合物对")
        print(f"   已过滤 {filtered_pairs_count} 个涉及载体分子的pair")  # ✅ 显示过滤数量

        return True

    def build_odor_network(self, max_depth=3, use_main_pairs=False):
        """Build network with ONLY odorous compounds"""
        print(f"Building odor-focused network (depth={max_depth}, use_main_pairs={use_main_pairs})...")

        # # Start with odorous compounds only
        # compounds_to_include = self.odorous_compounds.copy()
        # Start with odorous compounds only (排除载体分子)
        compounds_to_include = self.odorous_compounds.copy() - UBIQUITOUS_COMPOUNDS  # ✅ 过滤
        all_reactions_needed = set()

        print(f"  🎯 Starting with {len(compounds_to_include)} odorous compounds")
        print(f"     (已排除 {len(self.odorous_compounds & UBIQUITOUS_COMPOUNDS)} 个载体分子)")

        # Expand network to find intermediate compounds
        for depth in range(max_depth):
            print(f"  Expanding depth {depth + 1}...")

            new_reactions = set()
            for comp in sorted(compounds_to_include):
                if comp in self.compound_to_reactions:
                    new_reactions.update(self.compound_to_reactions[comp])

            all_reactions_needed.update(new_reactions)

            new_compounds = set()
            for reaction in sorted(new_reactions):
                if reaction in self.reaction_to_compounds:
                    reaction_compounds = self.reaction_to_compounds[reaction]
                    # # Only add compounds that connect odorous compounds
                    # new_compounds.update(reaction_compounds)
                    # ✅ 过滤载体分子
                    filtered_compounds = reaction_compounds - UBIQUITOUS_COMPOUNDS
                    new_compounds.update(filtered_compounds)

            old_size = len(compounds_to_include)
            compounds_to_include.update(new_compounds)

            print(f"    Added {len(new_compounds)} compounds, total: {len(compounds_to_include)}")

            if len(compounds_to_include) == old_size:
                break

        # Download EC data for reactions using the working v17 approach
        self.download_expanded_reaction_metadata(all_reactions_needed)

        # ✅ 条件加载 main-pair
        if use_main_pairs:
            self.load_main_pairs()
            print("  🔗 Using main-pair filtering for edge creation")
        else:
            print("  🔗 Using full reaction connectivity (no main-pair filtering)")

        # Build graph
        self.reaction_graph = nx.MultiDiGraph()

        # Add nodes
        for comp in compounds_to_include:
            # ✅ 双重检查：跳过载体分子
            if comp in UBIQUITOUS_COMPOUNDS:
                continue
            if comp in self.odorous_compounds:
                profile = self.compound_profiles[comp]
                self.reaction_graph.add_node(comp,
                                             node_type='odor_compound',
                                             name=profile['name'],
                                             odor_profile=profile['odor_profile'],
                                             dominant_odors=profile['dominant_odors']
                                             )
            else:
                self.reaction_graph.add_node(comp,
                                             node_type='intermediate',
                                             name=f"Intermediate_{comp}"
                                             )

        # # Add reaction nodes and edges
        # reactions_with_ec = 0
        # reactions_without_ec = 0
        #
        # for reaction in all_reactions_needed:
        #     reaction_node = f"R_{reaction}"
        #     ec_numbers = self.ec_data.get(reaction, [])
        #
        #     if ec_numbers:
        #         reactions_with_ec += 1
        #     else:
        #         reactions_without_ec += 1
        #
        #     self.reaction_graph.add_node(reaction_node,
        #                                  node_type='reaction',
        #                                  ec_numbers=ec_numbers
        #                                  )
        #
        #     # Add edges
        #     reactants = self.reaction_to_compounds[reaction] & compounds_to_include
        #     for comp in sorted(reactants):
        #         self.reaction_graph.add_edge(comp, reaction_node)
        #         self.reaction_graph.add_edge(reaction_node, comp)  # Assume reversible

        # ✅ 修改后：使用 main-pair 的边添加逻辑
        reactions_with_ec = 0
        reactions_without_ec = 0
        reactions_with_mainpair = 0
        reactions_without_mainpair = 0
        total_edges_added = 0

        for reaction in all_reactions_needed:
            reaction_node = f"R_{reaction}"
            ec_numbers = self.ec_data.get(reaction, [])

            if ec_numbers:
                reactions_with_ec += 1
            else:
                reactions_without_ec += 1

            self.reaction_graph.add_node(reaction_node,
                                         node_type='reaction',
                                         ec_numbers=ec_numbers)

            # # ✅ 方案A：优先使用 main-pair，无则不连（或极保守）
            # all_compounds = self.reaction_to_compounds[reaction] & compounds_to_include
            # edges_added_for_reaction = 0
            #
            # # 策略1：如果有 main-pair，严格使用
            # if reaction in self.main_pairs_index:
            #     reactions_with_mainpair += 1
            #
            #     for c_in, c_out_set in self.main_pairs_index[reaction].items():
            #         # 确保 c_in 在允许的化合物集合中
            #         if c_in not in all_compounds:
            #             continue
            #
            #         # 添加 C_in → R 边
            #         if not self.reaction_graph.has_edge(c_in, reaction_node):
            #             self.reaction_graph.add_edge(c_in, reaction_node)
            #             edges_added_for_reaction += 1
            #
            #         # 添加 R → C_out 边（main-pair 指定的产物）
            #         for c_out in c_out_set:
            #             if c_out == c_in:  # ✅ 自环保护
            #                 continue
            #             if c_out not in all_compounds:
            #                 continue
            #
            #             if not self.reaction_graph.has_edge(reaction_node, c_out):
            #                 self.reaction_graph.add_edge(reaction_node, c_out)
            #                 edges_added_for_reaction += 1
            # else:
            #     # 策略2：无 main-pair - 方案A是不连接，或极保守兜底
            #     reactions_without_mainpair += 1
            #
            #     # 🔴 方案A-strict：完全不连接
            #     pass  # 不添加任何边
            #
            #     # # 🟡 方案A-conservative：极保守兜底（可选）
            #     # # 只在化合物都是气味化合物时才连接
            #     # odorous_in_reaction = [c for c in all_compounds if c in self.odorous_compounds]
            #     #
            #     # if len(odorous_in_reaction) >= 2:
            #     #     # 只连接气味化合物之间（最多2对）
            #     #     for i, c1 in enumerate(odorous_in_reaction[:2]):
            #     #         self.reaction_graph.add_edge(c1, reaction_node)
            #     #         edges_added_for_reaction += 1
            #     #
            #     #         for c2 in odorous_in_reaction[i + 1:i + 2]:
            #     #             self.reaction_graph.add_edge(reaction_node, c2)
            #     #             edges_added_for_reaction += 1

            all_compounds = self.reaction_to_compounds[reaction] & compounds_to_include

            # ✅ 过滤载体分子
            all_compounds = all_compounds - UBIQUITOUS_COMPOUNDS

            edges_added_for_reaction = 0

            # ✅ 根据开关选择构图策略
            if use_main_pairs and reaction in self.main_pairs_index:
                # 策略A：使用 main-pair 精确连接
                reactions_with_mainpair += 1

                for c_in, c_out_set in self.main_pairs_index[reaction].items():
                    if c_in not in all_compounds:
                        continue

                    if not self.reaction_graph.has_edge(c_in, reaction_node):
                        self.reaction_graph.add_edge(c_in, reaction_node)
                        edges_added_for_reaction += 1

                    for c_out in c_out_set:
                        if c_out == c_in:  # 自环保护
                            continue
                        if c_out not in all_compounds:
                            continue

                        if not self.reaction_graph.has_edge(reaction_node, c_out):
                            self.reaction_graph.add_edge(reaction_node, c_out)
                            edges_added_for_reaction += 1

            elif not use_main_pairs:
                # 策略B：全连接（不使用 main-pair）
                reactions_without_mainpair += 1

                for comp in all_compounds:
                    # C → R
                    if not self.reaction_graph.has_edge(comp, reaction_node):
                        self.reaction_graph.add_edge(comp, reaction_node)
                        edges_added_for_reaction += 1

                    # R → C
                    if not self.reaction_graph.has_edge(reaction_node, comp):
                        self.reaction_graph.add_edge(reaction_node, comp)
                        edges_added_for_reaction += 1

            else:
                # 策略C：无 main-pair 且开启了过滤 → 不连接
                reactions_without_mainpair += 1
                pass

            total_edges_added += edges_added_for_reaction

        print(
            f"✓ Built network: {self.reaction_graph.number_of_nodes()} nodes, {self.reaction_graph.number_of_edges()} edges")
        print(f"  📊 Main-pair 统计:")
        print(f"     使用 main-pair: {reactions_with_mainpair} 反应")
        print(f"     无 main-pair: {reactions_without_mainpair} 反应")
        print(f"     Main-pair 覆盖率: {reactions_with_mainpair / len(all_reactions_needed) * 100:.1f}%")
        print(f"  Reactions with EC numbers: {reactions_with_ec}")
        print(f"  Reactions without EC numbers: {reactions_without_ec}")

        print(
            f"✓ Built network: {self.reaction_graph.number_of_nodes()} nodes, {self.reaction_graph.number_of_edges()} edges")
        print(f"  Reactions with EC numbers: {reactions_with_ec}")
        print(f"  Reactions without EC numbers: {reactions_without_ec}")

        # Show composition
        odor_nodes = sum(1 for n in self.reaction_graph.nodes()
                         if self.reaction_graph.nodes[n].get('node_type') == 'odor_compound')
        intermediate_nodes = sum(1 for n in self.reaction_graph.nodes()
                                 if self.reaction_graph.nodes[n].get('node_type') == 'intermediate')
        reaction_nodes = sum(1 for n in self.reaction_graph.nodes() if n.startswith('R_'))

        print(f"  Odor compounds: {odor_nodes}")
        print(f"  Intermediates: {intermediate_nodes}")
        print(f"  Reactions: {reaction_nodes}")

        return True

    def _save_pathways_cache(self, pathways, max_length, max_paths_per_pair):
        """Save computed pathways to simple cache file"""
        try:
            cache_file = self.cache_dir / "pathways_cache.pkl"

            cache_data = {
                'pathways': pathways,
                'parameters': {
                    'max_length': max_length,
                    'max_paths_per_pair': max_paths_per_pair
                },
                'metadata': {
                    'created_time': time.time(),
                    'pathway_count': len(pathways),
                    'odor_compounds_count': len(self.odorous_compounds)
                }
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            file_size = cache_file.stat().st_size
            print(f"  💾 Pathways cached: {len(pathways)} pathways, {file_size:,} bytes")
            print(f"     Cache file: pathways_cache.pkl")

            return True

        except Exception as e:
            print(f"  ❌ Failed to cache pathways: {e}")
            return False

    def _load_pathways_cache(self, max_length, max_paths_per_pair):
        """Load pathways from simple cache file"""
        try:
            cache_file = self.cache_dir / "pathways_cache.pkl"

            if not cache_file.exists():
                return None

            print(f"  📦 Loading pathways from cache: pathways_cache.pkl")

            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            pathways = cache_data['pathways']
            metadata = cache_data.get('metadata', {})
            params = cache_data.get('parameters', {})

            print(f"  ✅ Cache loaded successfully:")
            print(f"     Pathways: {len(pathways)}")
            print(f"     Created: {time.ctime(metadata.get('created_time', 0))}")
            print(
                f"     Cached parameters: max_length={params.get('max_length')}, max_paths_per_pair={params.get('max_paths_per_pair')}")
            print(f"     Current parameters: max_length={max_length}, max_paths_per_pair={max_paths_per_pair}")

            return pathways

        except Exception as e:
            print(f"  ❌ Failed to load pathway cache: {e}")
            return None

    def find_odor_pathways(self, max_length=5, max_paths_per_pair=1, use_cache=True, force_recompute=False):
        """Enhanced pathway finding with caching support"""
        print(f"Finding pathways ≤{max_length} reaction steps, max {max_paths_per_pair} paths per pair")

        # Try to load from cache first
        if use_cache and not force_recompute:
            print("🔍 Checking for cached pathways...")
            cached_pathways = self._load_pathways_cache(max_length, max_paths_per_pair)
            if cached_pathways is not None:
                self.pathways = cached_pathways
                print(f"✅ Using cached pathways: {len(cached_pathways)} pathways loaded")
                return cached_pathways

        print("🚀 Computing pathways (this may take a while)...")

        odor_compounds = sorted(self.odorous_compounds)
        total_possible_pairs = len(odor_compounds) * (len(odor_compounds) - 1) // 2
        print(f"  Total compounds: {len(odor_compounds)}")
        print(f"  Total possible pairs: {total_possible_pairs:,}")

        # 🔄 Build optimized graph with proper weights
        print("  🔄 Building optimized graph with reaction step weights...")
        simple_graph = self._create_simple_graph_for_pathfinding()

        # 🚀 Fixed batch optimization: single-source Dijkstra per source to avoid confusion
        print("  📊 Pre-computing reachable pairs with single-source Dijkstra...")
        start_time = time.time()

        reachable_pairs = set()
        for src in tqdm(odor_compounds, desc="Computing distances"):
            try:
                # Use single-source Dijkstra with cutoff = max_length (since weights now represent reaction steps)
                distances = nx.single_source_dijkstra_path_length(
                    simple_graph, src, cutoff=max_length
                )

                # Add reachable target compounds
                for tgt, dist in distances.items():
                    if (tgt in self.odorous_compounds and
                            src < tgt and  # Avoid duplicates (src,tgt) vs (tgt,src)
                            dist <= max_length):
                        reachable_pairs.add((src, tgt))

            except Exception as e:
                # Skip sources that have no outgoing paths
                continue

        reachable_pairs = sorted(reachable_pairs)
        batch_time = time.time() - start_time

        print(f"  ✓ Pre-computation done in {batch_time:.1f}s")
        print(f"  ✓ Found {len(reachable_pairs):,} reachable pairs (≤{max_length} steps)")
        print(f"  ✓ Filtered out {total_possible_pairs - len(reachable_pairs):,} unreachable pairs")

        # 🎯 Apply Yen k-shortest paths only to reachable pairs
        pathways = []
        pairs_with_paths = 0
        total_paths_found = 0

        print(f"  🔍 Finding {max_paths_per_pair} paths per reachable pair...")
        for source, target in tqdm(reachable_pairs, desc="Yen k‑shortest"):
            try:
                # Find multiple paths for this pair
                pair_pathways = self._find_multiple_paths_optimized(
                    source, target, max_length, max_paths_per_pair, simple_graph
                )

                if pair_pathways:
                    pairs_with_paths += 1
                    total_paths_found += len(pair_pathways)
                    pathways.extend(pair_pathways)

            except Exception:
                continue

        self.pathways = pathways

        # Save to cache if successful
        # if pathways and use_cache:
        if pathways:
            print("💾 Saving pathways to cache...")
            self._save_pathways_cache(pathways, max_length, max_paths_per_pair)

        # Enhanced statistics
        connectivity_rate = pairs_with_paths / len(reachable_pairs) * 100 if reachable_pairs else 0
        avg_paths_per_connected_pair = total_paths_found / pairs_with_paths if pairs_with_paths > 0 else 0

        print(f"\n✓ PATHWAY DISCOVERY COMPLETE")
        print(f"  Total pathways found: {len(pathways):,}")
        print(f"  Connected pairs: {pairs_with_paths:,}/{len(reachable_pairs):,} ({connectivity_rate:.1f}%)")
        print(f"  Average paths per connected pair: {avg_paths_per_connected_pair:.1f}")
        print(f"  Training sample diversity: {avg_paths_per_connected_pair:.1f}x increase")

        return pathways

    def _create_simple_graph_for_pathfinding(self):
        """
        把 MultiDiGraph 精简成 DiGraph，并让"一次完整反应"权重为 1。
        ── 方案：C→R 与 R→C 各给 0.5 权重，保证路径权重 = 反应步数。
        仅保留化合物↔反应的交替边，避免出现 C→C 或 R→R。
        """
        simple_graph = nx.DiGraph()

        # Add all nodes first
        for node, data in self.reaction_graph.nodes(data=True):
            simple_graph.add_node(node, **data)

        # Add edges with proper weights: C→R and R→C each get weight 0.5
        # This makes one complete reaction (C→R→C) have total weight 1.0
        for u, v in self.reaction_graph.edges():
            # Ensure alternating structure: one end must be Reaction, other must be Compound
            if u.startswith('R_') == v.startswith('R_'):
                continue  # Skip R→R or C→C edges

            if not simple_graph.has_edge(u, v):
                simple_graph.add_edge(u, v, weight=0.5)  # C→R or R→C

        return simple_graph

    def _find_multiple_paths_optimized(self, source, target, max_length, max_paths_per_pair, simple_graph):
        """Optimized multiple path finding using Yen algorithm with proper weight handling"""
        found_paths = []

        # 🔒 Safety check: ensure source and target are compounds, not reactions
        assert not source.startswith('R_') and not target.startswith('R_'), \
            f"Source/target must be compounds, got: {source}, {target}"

        try:
            # 🚀 Use Yen's algorithm with PROPER WEIGHT parameter (critical fix!)
            from networkx.algorithms.simple_paths import shortest_simple_paths

            try:
                # ✅ FIXED: Use weight='weight' to respect our 0.5/0.5 weight design
                path_generator = shortest_simple_paths(simple_graph, source, target, weight='weight')

                paths_collected = 0
                for path in path_generator:
                    reaction_steps = sum(1 for node in path if node.startswith('R_'))
                    if reaction_steps <= max_length:
                        # Parse pathway using original MultiDiGraph to get proper EC data
                        pathway_info = self._parse_pathway(path, source, target)
                        if pathway_info:
                            pathway_info['path_type'] = 'yen_shortest' if paths_collected == 0 else 'yen_alternative'
                            pathway_info['path_rank'] = paths_collected + 1
                            found_paths.append(pathway_info)

                            paths_collected += 1
                            if paths_collected >= max_paths_per_pair:
                                break
                    else:
                        # If we hit a path longer than max_length, all subsequent paths will be longer
                        break

            except Exception as yen_error:
                print(f"    ⚠️ Yen algorithm failed for {source}->{target}: {yen_error}")
                # Fallback to single shortest path with weight
                try:
                    path = nx.shortest_path(simple_graph, source, target, weight='weight')
                    reaction_steps = sum(1 for node in path if node.startswith('R_'))
                    if reaction_steps <= max_length:
                        pathway_info = self._parse_pathway(path, source, target)
                        if pathway_info:
                            pathway_info['path_type'] = 'shortest_weighted'
                            pathway_info['path_rank'] = 1
                            found_paths.append(pathway_info)
                except nx.NetworkXNoPath:
                    pass

        except nx.NetworkXNoPath:
            pass
        except Exception as e:
            # Final fallback: try with original graph
            try:
                path = nx.shortest_path(self.reaction_graph, source, target)
                reaction_steps = sum(1 for node in path if node.startswith('R_'))
                if reaction_steps <= max_length:
                    pathway_info = self._parse_pathway(path, source, target)
                    if pathway_info:
                        pathway_info['path_type'] = 'shortest_fallback'
                        pathway_info['path_rank'] = 1
                        found_paths.append(pathway_info)
            except Exception:
                pass

        return found_paths

    def _parse_pathway(self, path, source, target):
        """Parse pathway into structured format with robust path validation"""

        # 🔒 Safety check: ensure path starts with a compound (not reaction)
        if not path or path[0].startswith('R_'):
            print(f"    ⚠️ Invalid path structure: starts with reaction node {path[0] if path else 'empty'}")
            return None

        # 🔒 Safety check: ensure alternating C-R-C pattern
        for i in range(len(path) - 1):
            current_is_reaction = path[i].startswith('R_')
            next_is_reaction = path[i + 1].startswith('R_')
            if current_is_reaction == next_is_reaction:
                print(f"    ⚠️ Invalid path structure: non-alternating pattern at {i}-{i + 1}")
                return None

        pathway_steps = []
        ec_sequence = []

        for i in range(0, len(path) - 1, 2):
            if i + 2 < len(path):
                comp1 = path[i]
                reaction = path[i + 1]
                comp2 = path[i + 2]

                reaction_info = self.reaction_graph.nodes[reaction]
                ec_numbers = reaction_info.get('ec_numbers', [])

                # # # 🔧 FIXED: Handle reactions without EC numbers by assigning special token
                # # if ec_sections:
                # #     ec_sequence.extend(ec_sections[:1])  # Take first EC section
                # # else:
                # #     # Instead of discarding, assign a special "unknown EC" token
                # #     ec_sequence.append('EC:unknown')
                # # ✅ 动态截取到三级
                # if ec_numbers:
                #     ec3 = self._to_ec3(ec_numbers[0])  # 取第一个，截取到三级
                #     if ec3:
                #         ec_sequence.append(ec3)
                #     else:
                #         ec_sequence.append('unknown')
                # else:
                #     ec_sequence.append('unknown')
                #
                # pathway_steps.append({
                #     'from': comp1,
                #     'reaction': reaction.replace('R_', ''),
                #     'to': comp2,
                #     'ec_sections': ec_sections,
                #     'ec_numbers': ec_numbers,
                #     'has_ec_data': len(ec_sections) > 0
                # })
                # ✅ 动态截取到三级
                if ec_numbers:
                    ec3 = to_ec_level(ec_numbers[0], level=3)  # 取第一个 EC，截取到三级
                    if ec3:
                        ec_sequence.append(ec3)
                    else:
                        ec_sequence.append('unknown')
                else:
                    ec_sequence.append('unknown')

                pathway_steps.append({
                    'from': comp1,
                    'reaction': reaction.replace('R_', ''),
                    'to': comp2,
                    'ec_numbers': ec_numbers,
                    'has_ec_data': len(ec_numbers) > 0  # ✅ 检查 ec_numbers
                })

        # 🔧 FIXED: Don't discard pathways without EC - they may still be valuable
        # Only discard if pathway is too short or invalid
        if len(pathway_steps) == 0:
            return None

        # Calculate odor similarity
        source_profile = self.compound_profiles[source]['odor_profile']
        target_profile = self.compound_profiles[target]['odor_profile']

        similarity = np.dot(source_profile, target_profile) / (
                np.linalg.norm(source_profile) * np.linalg.norm(target_profile) + 1e-10)

        pathway_info = {
            'source': source,
            'target': target,
            'source_odors': self.compound_profiles[source]['dominant_odors'],
            'target_odors': self.compound_profiles[target]['dominant_odors'],
            'pathway_length': len(pathway_steps),
            'ec_sequence': ec_sequence,
            'odor_similarity': similarity,
            'steps': pathway_steps,
            'has_unknown_ec': 'unknown' in ec_sequence,
            'ec_coverage': sum(1 for step in pathway_steps if step['has_ec_data']) / len(
                pathway_steps) if pathway_steps else 0
        }

        return pathway_info

    def list_pathway_caches(self):
        """List available pathway cache file"""
        cache_file = self.cache_dir / "pathways_cache.pkl"

        if not cache_file.exists():
            print("No pathway cache file found.")
            return []

        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            metadata = cache_data.get('metadata', {})
            params = cache_data.get('parameters', {})

            info = {
                'file': cache_file.name,
                'pathways': metadata.get('pathway_count', 0),
                'created': time.ctime(metadata.get('created_time', 0)),
                'max_length': params.get('max_length', '?'),
                'max_paths_per_pair': params.get('max_paths_per_pair', '?'),
                'size_mb': cache_file.stat().st_size / (1024 * 1024)
            }

            print(f"Found pathway cache file:")
            print(f"  📦 {info['file']}")
            print(f"     Pathways: {info['pathways']:,}")
            print(f"     Parameters: max_length={info['max_length']}, max_paths_per_pair={info['max_paths_per_pair']}")
            print(f"     Created: {info['created']}")
            print(f"     Size: {info['size_mb']:.1f} MB")

            return [info]

        except Exception as e:
            print(f"  ❌ pathways_cache.pkl: Error reading cache - {e}")
            return []

    def clear_pathway_cache(self):
        """Delete the pathway cache file"""
        cache_file = self.cache_dir / "pathways_cache.pkl"

        if cache_file.exists():
            try:
                cache_file.unlink()
                print("✅ Pathway cache cleared (pathways_cache.pkl deleted)")
                return True
            except Exception as e:
                print(f"❌ Failed to clear cache: {e}")
                return False
        else:
            print("No pathway cache file to clear.")
            return True


def to_ec_level(ec: str, level: int = 3) -> str:
    """
    截取 EC 编号到指定级别
    【从 paste1 第35-62行复制】
    """
    if not ec or '.' not in ec:
        return ''
    parts = ec.split('.')
    if len(parts) < level:
        return ''
    result_parts = parts[:level]
    if result_parts[-1] == '-':
        return ''
    return '.'.join(result_parts)


# 【从 paste1 第65-620行复制整个 OdorPathwayAnalyzer 类】
# class OdorPathwayAnalyzer:
#     """Optimized analyzer with pathway caching for debugging"""
#     ... (完整复制)


# =============================================================================
# PART 2: 笛卡尔展开部分 - 从 paste2 复制
# =============================================================================

def expand_to_odor_events(pathways: List[Dict]) -> List[Dict]:
    """
    展开为气味级事件（笛卡尔积 + 权重分摊）
    【从 paste2 第24-60行复制，略作调整】
    """
    events = []

    for pw in pathways:
        # 提取EC3序列
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

        n_source = len(source_odors)
        n_target = len(target_odors)

        if n_source == 0 or n_target == 0:
            continue

        weight = 1.0 / (n_source * n_target)

        # 笛卡尔积展开
        for s_odor in source_odors:
            for t_odor in target_odors:
                events.append({
                    'source_odor': s_odor,
                    'target_odor': t_odor,
                    'ec_sequence': tuple(ec3_seq),
                    'source_compound': pw['source'],
                    'target_compound': pw['target'],
                    'weight': weight
                })

    return events


def build_odor_pattern_index(events: List[Dict]) -> Dict:
    """
    构建气味pattern索引（加权）
    【从 paste2 第63-120行的 analyze_odor_patterns 改编】
    """
    # 核心索引
    triplet_counts = Counter()  # (src, ec_seq, tgt) → weight
    triplet_examples = defaultdict(list)

    pair_to_chains = defaultdict(Counter)  # (src, tgt) → {ec_seq: weight}
    source_to_targets = defaultdict(Counter)  # src → {tgt: weight}
    chain_to_pairs = defaultdict(Counter)  # ec_seq → {(src, tgt): weight}

    ec_seq_counts = Counter()  # ec_seq → total_weight

    for event in events:
        s = event['source_odor']
        t = event['target_odor']
        ec = event['ec_sequence']
        w = event['weight']

        triplet_counts[(s, ec, t)] += w
        pair_to_chains[(s, t)][ec] += w
        source_to_targets[s][t] += w
        chain_to_pairs[ec][(s, t)] += w
        ec_seq_counts[ec] += w

        # 保存样例
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


# =============================================================================
# PART 3: 复杂一阶逻辑规则提取 - 替换原有简单规则
# =============================================================================

@dataclass
class ComplexFOLRule:
    """
    复杂一阶逻辑规则的数据结构

    支持的规则类型：
    - necessary: 必要条件 ∀P: transforms(S,T,P) → via_ec(P, E)
    - sufficient: 充分条件 via_ec(P,E1) ∧ via_ec(P,E2) → produces(P, T)
    - exclusion: 排斥规则 from(S) ∧ via_ec(P,E) → ¬to(T)
    - disjunctive_source: 析取源 (from(S1) ∨ from(S2)) ∧ via(E) → to(T)
    - mutual_exclusion: EC互斥 via(E1) ∧ via(E2) → ⊥ (罕见共现)
    - conditional_necessary: 条件必要 from(S) ∧ to(T) → via(E)
    """
    rule_type: str
    quantifier: str  # 'forall', 'exists', 'none'
    head: Dict  # 规则头（结论）
    body: Dict  # 规则体（前提）
    support: float
    confidence: float
    coverage: float  # 规则覆盖的样本比例
    examples: List[Dict] = field(default_factory=list)

    def to_fol_string(self) -> str:
        """转换为标准一阶逻辑字符串"""

        if self.rule_type == 'necessary':
            # ∀P: transforms(S, T, P) → via_ec(P, E)
            src, tgt = self.body['source'], self.body['target']
            ec = self.head['ec']
            return f"∀P: transforms({src}, {tgt}, P) → via_ec(P, {ec})"

        elif self.rule_type == 'sufficient':
            # via_ec(P, E1) ∧ via_ec(P, E2) ∧ ... → produces(P, T)
            ecs = self.body['ec_set']
            tgt = self.head['target']
            ec_conds = ' ∧ '.join([f"via_ec(P, {e})" for e in ecs])
            return f"{ec_conds} → produces(P, {tgt})"

        elif self.rule_type == 'exclusion':
            # from_odor(P, S) ∧ via_ec(P, E) → ¬to_odor(P, T)
            src = self.body.get('source', '*')
            ec_seq = self.body['ec_sequence']
            tgt = self.head['excluded_target']
            ec_str = ' → '.join(ec_seq) if isinstance(ec_seq, (list, tuple)) else ec_seq
            return f"from_odor(P, {src}) ∧ via_ec(P, [{ec_str}]) → ¬to_odor(P, {tgt})"

        elif self.rule_type == 'disjunctive_source':
            # (from(S1) ∨ from(S2)) ∧ via(E) → to(T)
            sources = self.body['sources']
            ec_seq = self.body['ec_sequence']
            tgt = self.head['target']
            src_disj = ' ∨ '.join([f"from({s})" for s in sources])
            ec_str = ' → '.join(ec_seq)
            return f"({src_disj}) ∧ via([{ec_str}]) → to({tgt})"

        elif self.rule_type == 'mutual_exclusion':
            # via(E1) ∧ via(E2) → ⊥
            ec1, ec2 = self.body['ec1'], self.body['ec2']
            ctx = self.body.get('context', '')
            return f"via_ec(P, {ec1}) ∧ via_ec(P, {ec2}) → ⊥  {ctx}"

        elif self.rule_type == 'conditional_necessary':
            # from(S) ∧ to(T) → must_via(E)
            src, tgt = self.body['source'], self.body['target']
            ec = self.head['necessary_ec']
            return f"from({src}) ∧ to({tgt}) → must_via({ec})"

        elif self.rule_type == 'ec_class_rule':
            # 基于EC大类的规则
            ec_class = self.body['ec_class']
            effect = self.head['effect']
            return f"via_ec_class(P, {ec_class}) → {effect}"

        return f"[{self.rule_type}] {self.body} → {self.head}"

    def to_prolog(self) -> str:
        """转换为Prolog风格Horn子句"""

        if self.rule_type == 'necessary':
            src, tgt = self.body['source'], self.body['target']
            ec = self.head['ec']
            return f"must_via(P, '{ec}') :- transforms(P, '{src}', '{tgt}')."

        elif self.rule_type == 'sufficient':
            ecs = self.body['ec_set']
            tgt = self.head['target']
            ec_conds = ', '.join([f"via_ec(P, '{e}')" for e in ecs])
            return f"produces(P, '{tgt}') :- {ec_conds}."

        elif self.rule_type == 'exclusion':
            src = self.body.get('source', '_')
            ec_seq = self.body['ec_sequence']
            tgt = self.head['excluded_target']
            return f"impossible(P, '{tgt}') :- from(P, '{src}'), via_seq(P, {list(ec_seq)})."

        elif self.rule_type == 'conditional_necessary':
            src, tgt = self.body['source'], self.body['target']
            ec = self.head['necessary_ec']
            return f"required_ec('{src}', '{tgt}', '{ec}')."

        return f"% {self.rule_type}: {self.to_fol_string()}"


class ComplexRuleExtractor:
    """
    复杂一阶逻辑规则提取器

    提取规则类型：
    1. 必要条件：某转化必须经过的EC
    2. 充分条件：高置信度的EC组合→目标
    3. 排斥规则：某EC序列永不产生的气味
    4. 析取源规则：多个源气味共享的转化模式
    5. EC互斥：很少共现的EC对
    6. 条件必要：特定转化的必经EC
    """

    def __init__(self, pattern_index: Dict,
                 min_support: float = 2.0,
                 min_confidence: float = 0.1,
                 high_confidence: float = 0.8,
                 exclusion_threshold: float = 0.95):
        """
        Args:
            pattern_index: build_odor_pattern_index() 的输出
            min_support: 最小支持度
            min_confidence: 最小置信度
            high_confidence: 高置信度阈值（用于充分条件）
            exclusion_threshold: 排斥阈值（未出现比例）
        """
        self.index = pattern_index
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.high_confidence = high_confidence
        self.exclusion_threshold = exclusion_threshold

        self.rules: List[ComplexFOLRule] = []

        # 预计算统计量
        self._precompute_statistics()

    def _precompute_statistics(self):
        """预计算用于规则提取的统计量"""
        triplet_counts = self.index['triplet_counts']

        # 1. 每个(src, tgt)对的所有EC序列
        self.pair_ec_sets = defaultdict(list)  # (src, tgt) → [ec_seq1, ec_seq2, ...]
        self.pair_ec_weights = defaultdict(lambda: defaultdict(float))  # (src, tgt) → {ec_seq: weight}

        # 2. 每个EC序列的所有(src, tgt)对
        self.ec_to_pairs = defaultdict(lambda: defaultdict(float))  # ec_seq → {(src, tgt): weight}

        # 3. 每个EC序列产生的目标气味
        self.ec_to_targets = defaultdict(Counter)  # ec_seq → {target: weight}

        # 4. 每个源气味使用的EC序列
        self.source_ec_usage = defaultdict(Counter)  # source → {ec_seq: weight}

        # 5. EC共现统计
        self.ec_cooccurrence = Counter()  # (ec1, ec2) → count
        self.ec_total_usage = Counter()  # ec → total_count

        # 6. 所有目标气味集合
        self.all_targets = set()

        for (src, ec_seq, tgt), weight in triplet_counts.items():
            # 更新pair→EC映射
            self.pair_ec_sets[(src, tgt)].append(ec_seq)
            self.pair_ec_weights[(src, tgt)][ec_seq] += weight

            # 更新EC→pair映射
            self.ec_to_pairs[ec_seq][(src, tgt)] += weight

            # 更新EC→target映射
            self.ec_to_targets[ec_seq][tgt] += weight

            # 更新source→EC映射
            self.source_ec_usage[src][ec_seq] += weight

            # 更新EC共现
            for i, ec in enumerate(ec_seq):
                self.ec_total_usage[ec] += weight
                for j in range(i + 1, len(ec_seq)):
                    pair = tuple(sorted([ec_seq[i], ec_seq[j]]))
                    self.ec_cooccurrence[pair] += weight

            self.all_targets.add(tgt)

        # 全局target频率（用于排斥规则筛选）
        self.target_global_counts = Counter()
        for (src, ec_seq, tgt), w in triplet_counts.items():
            self.target_global_counts[tgt] += w
        self.total_triplet_weight = sum(triplet_counts.values())

        print(f"  预计算完成: {len(self.pair_ec_sets)} 对转化, {len(self.ec_to_pairs)} 种EC序列")

    def extract_all_rules(self) -> List[ComplexFOLRule]:
        """提取所有类型的复杂规则"""
        print("=" * 80)
        print("复杂一阶逻辑规则提取")
        print("=" * 80)

        self.rules = []

        # 1. 必要条件规则
        print("\n🔷 提取必要条件规则 (Necessary Condition)...")
        print("   ∀P: transforms(S, T, P) → via_ec(P, E)")
        necessary_rules = self._extract_necessary_rules()
        self.rules.extend(necessary_rules)
        print(f"   ✓ 提取了 {len(necessary_rules)} 条必要条件规则")

        # 2. 充分条件规则（高置信度）
        print("\n🔷 提取充分条件规则 (Sufficient Condition)...")
        print("   via_ec(P, E1) ∧ via_ec(P, E2) → produces(P, T)  [conf > {:.0%}]".format(self.high_confidence))
        sufficient_rules = self._extract_sufficient_rules()
        self.rules.extend(sufficient_rules)
        print(f"   ✓ 提取了 {len(sufficient_rules)} 条充分条件规则")

        # 3. 排斥规则
        print("\n🔷 提取排斥规则 (Exclusion)...")
        print("   from(S) ∧ via(E) → ¬to(T)")
        exclusion_rules = self._extract_exclusion_rules()
        self.rules.extend(exclusion_rules)
        print(f"   ✓ 提取了 {len(exclusion_rules)} 条排斥规则")

        # 4. 析取源规则
        print("\n🔷 提取析取源规则 (Disjunctive Source)...")
        print("   (from(S1) ∨ from(S2)) ∧ via(E) → to(T)")
        disjunctive_rules = self._extract_disjunctive_source_rules()
        self.rules.extend(disjunctive_rules)
        print(f"   ✓ 提取了 {len(disjunctive_rules)} 条析取源规则")

        # 5. EC互斥规则
        print("\n🔷 提取EC互斥规则 (Mutual Exclusion)...")
        print("   via(E1) ∧ via(E2) → ⊥")
        mutex_rules = self._extract_mutual_exclusion_rules()
        self.rules.extend(mutex_rules)
        print(f"   ✓ 提取了 {len(mutex_rules)} 条EC互斥规则")

        # 6. 条件必要规则
        print("\n🔷 提取条件必要规则 (Conditional Necessary)...")
        print("   from(S) ∧ to(T) → must_via(E)")
        cond_necessary_rules = self._extract_conditional_necessary_rules()
        self.rules.extend(cond_necessary_rules)
        print(f"   ✓ 提取了 {len(cond_necessary_rules)} 条条件必要规则")

        # 按类型和支持度排序
        self.rules.sort(key=lambda r: (r.rule_type, -r.support))

        print(f"\n{'=' * 80}")
        print(f"✅ 总计提取 {len(self.rules)} 条复杂一阶逻辑规则")
        print(f"{'=' * 80}")

        return self.rules

    def _extract_necessary_rules(self) -> List[ComplexFOLRule]:
        """
        提取必要条件规则

        对于每个(source, target)对，找出所有路径都必须经过的EC
        即EC序列的交集
        """
        rules = []

        for (src, tgt), ec_sequences in self.pair_ec_sets.items():
            if len(ec_sequences) < 2:
                continue  # 需要多条路径才能判断必要性

            # 将每个EC序列转为集合
            ec_sets = [set(seq) for seq in ec_sequences]

            # 计算交集 - 所有路径都经过的EC
            common_ecs = ec_sets[0].copy()
            for ec_set in ec_sets[1:]:
                common_ecs &= ec_set

            # 过滤掉背景EC（太常见，缺乏区分度）
            common_ecs -= BACKGROUND_EC

            if not common_ecs:
                continue

            # 计算支持度（该转化对的总权重）
            total_weight = sum(self.pair_ec_weights[(src, tgt)].values())

            if total_weight < self.min_support:
                continue

            # 为每个必要EC创建规则
            for ec in common_ecs:
                rule = ComplexFOLRule(
                    rule_type='necessary',
                    quantifier='forall',
                    head={'ec': ec},
                    body={'source': src, 'target': tgt},
                    support=total_weight,
                    confidence=1.0,  # 交集意味着100%置信度
                    coverage=len(ec_sequences) / len(self.pair_ec_sets),
                    examples=[{'paths': len(ec_sequences)}]
                )
                rules.append(rule)

        rules.sort(key=lambda r: r.support, reverse=True)
        return rules

    def _extract_sufficient_rules(self) -> List[ComplexFOLRule]:
        """
        提取充分条件规则

        找出某些EC组合高置信度地导向特定目标气味
        """
        rules = []

        # 分析每个EC序列
        for ec_seq, target_counts in self.ec_to_targets.items():
            total_usage = sum(target_counts.values())

            if total_usage < self.min_support:
                continue

            # 检查是否有高置信度的目标
            for target, target_weight in target_counts.items():
                confidence = target_weight / total_usage

                if confidence >= self.high_confidence:
                    # 将EC序列作为集合（忽略顺序）
                    ec_set = tuple(sorted(set(ec_seq)))

                    rule = ComplexFOLRule(
                        rule_type='sufficient',
                        quantifier='none',
                        head={'target': target},
                        body={'ec_set': ec_set, 'ec_sequence': ec_seq},
                        support=target_weight,
                        confidence=confidence,
                        coverage=target_weight / total_usage
                    )
                    rules.append(rule)

        # 去重：相同ec_set和target只保留最高置信度的
        unique_rules = {}
        for rule in rules:
            key = (rule.body['ec_set'], rule.head['target'])
            if key not in unique_rules or rule.confidence > unique_rules[key].confidence:
                unique_rules[key] = rule

        rules = list(unique_rules.values())
        rules.sort(key=lambda r: (r.confidence, r.support), reverse=True)
        return rules

    def _extract_exclusion_rules(self) -> List[ComplexFOLRule]:
        """
        提取排斥规则（优化版）

        只有当 target 在全局上"足够常见"，但在某 EC 序列下"完全不出现"时，
        才认为是真正的排斥关系。
        """
        rules = []

        # 全局target频率阈值：至少占总权重的1%
        min_global_ratio = 0.01
        min_global_weight = self.total_triplet_weight * min_global_ratio

        # 常见target集合（只对这些target检测排斥）
        common_targets = {
            tgt for tgt, w in self.target_global_counts.items()
            if w >= min_global_weight
        }

        print(f"     常见target数量: {len(common_targets)} (全局占比≥{min_global_ratio:.0%})")

        for ec_seq, target_counts in self.ec_to_targets.items():
            ec_total = sum(target_counts.values())

            # EC序列本身要足够常见
            if ec_total < self.min_support * 5:
                continue

            # 获取代表性source
            source_counts = Counter()
            for (src, tgt), weight in self.ec_to_pairs[ec_seq].items():
                source_counts[src] += weight

            if not source_counts:
                continue

            top_source = source_counts.most_common(1)[0][0]

            # 该EC序列实际产生的target
            produced_targets = set(target_counts.keys())

            # 只检查"常见但未出现"的target
            excluded_targets = common_targets - produced_targets

            for excluded_tgt in excluded_targets:
                global_weight = self.target_global_counts[excluded_tgt]

                # 计算"期望 vs 实际"的偏离程度
                # 如果这个target在全局占5%，但在这个EC序列下是0%，说明强排斥
                expected_ratio = global_weight / self.total_triplet_weight
                actual_ratio = 0.0  # 完全未出现

                # 排斥强度 = 期望占比（期望越高，排斥越显著）
                exclusion_strength = expected_ratio

                rule = ComplexFOLRule(
                    rule_type='exclusion',
                    quantifier='forall',
                    head={'excluded_target': excluded_tgt},
                    body={'source': top_source, 'ec_sequence': ec_seq},
                    support=ec_total,
                    confidence=1.0,  # 完全未出现
                    coverage=exclusion_strength,  # 用期望占比表示显著性
                    examples=[{'global_ratio': f"{expected_ratio:.1%}"}]
                )
                rules.append(rule)

        # 按 support * coverage 排序（既要EC常见，又要target本身常见）
        rules.sort(key=lambda r: r.support * r.coverage, reverse=True)

        # return rules[:200]
        return rules

    def _extract_disjunctive_source_rules(self) -> List[ComplexFOLRule]:
        """
        提取析取源规则

        找出多个源气味通过相同EC序列到达相同目标的模式
        (from(S1) ∨ from(S2)) ∧ via(EC) → to(T)
        """
        rules = []

        # 按(ec_seq, target)分组，找共享的sources
        ec_target_sources = defaultdict(lambda: defaultdict(float))  # (ec, tgt) → {src: weight}

        triplet_counts = self.index['triplet_counts']
        for (src, ec_seq, tgt), weight in triplet_counts.items():
            ec_target_sources[(ec_seq, tgt)][src] += weight

        for (ec_seq, tgt), source_weights in ec_target_sources.items():
            # 需要至少2个不同的源
            if len(source_weights) < 2:
                continue

            total_weight = sum(source_weights.values())
            if total_weight < self.min_support:
                continue

            # 取权重最高的几个源
            top_sources = [src for src, _ in Counter(source_weights).most_common(5)]

            if len(top_sources) >= 2:
                rule = ComplexFOLRule(
                    rule_type='disjunctive_source',
                    quantifier='exists',
                    head={'target': tgt},
                    body={'sources': top_sources, 'ec_sequence': ec_seq},
                    support=total_weight,
                    confidence=1.0,
                    coverage=len(top_sources) / len(self.source_ec_usage)
                )
                rules.append(rule)

        rules.sort(key=lambda r: (len(r.body['sources']), r.support), reverse=True)
        # return rules[:200]
        return rules

    def _extract_mutual_exclusion_rules(self) -> List[ComplexFOLRule]:
        """
        提取EC互斥规则

        找出很少在同一路径中共现的EC对
        """
        rules = []

        # 计算每对EC的期望共现次数 vs 实际共现次数
        total_paths = sum(self.index['ec_seq_counts'].values())

        ec_list = [ec for ec, count in self.ec_total_usage.items() if count >= self.min_support]

        for i, ec1 in enumerate(ec_list):
            for ec2 in ec_list[i + 1:]:
                pair = tuple(sorted([ec1, ec2]))

                # 实际共现次数
                actual_cooccur = self.ec_cooccurrence.get(pair, 0)

                # 期望共现次数（独立假设）
                p1 = self.ec_total_usage[ec1] / total_paths
                p2 = self.ec_total_usage[ec2] / total_paths
                expected_cooccur = p1 * p2 * total_paths

                # 如果实际远小于期望，说明互斥
                if expected_cooccur > 5:  # 确保期望足够大
                    ratio = actual_cooccur / expected_cooccur

                    if ratio < 0.1:  # 实际不到期望的10%
                        rule = ComplexFOLRule(
                            rule_type='mutual_exclusion',
                            quantifier='forall',
                            head={'contradiction': True},
                            body={'ec1': ec1, 'ec2': ec2,
                                  'context': f"(ratio={ratio:.2f})"},
                            support=expected_cooccur,
                            confidence=1 - ratio,
                            coverage=ratio
                        )
                        rules.append(rule)

        rules.sort(key=lambda r: r.confidence, reverse=True)
        # return rules[:100]
        return rules

    def _extract_conditional_necessary_rules(self) -> List[ComplexFOLRule]:
        """
        提取条件必要规则

        对于特定(source, target)转化，找出必须经过的EC
        与necessary不同，这里强调条件性
        """
        rules = []

        for (src, tgt), ec_weights in self.pair_ec_weights.items():
            total_weight = sum(ec_weights.values())

            if total_weight < self.min_support:
                continue

            # 统计每个EC在该转化中的出现频率
            ec_frequency = Counter()
            for ec_seq, weight in ec_weights.items():
                for ec in ec_seq:
                    ec_frequency[ec] += weight

            # # 找出高频EC（出现在>80%的路径中）
            # for ec, ec_weight in ec_frequency.items():
            #     coverage = ec_weight / total_weight
            #
            #     if coverage >= 0.8:  # 80%以上的路径都经过
            # 找出高频EC（出现在>80%的路径中）
            for ec, ec_weight in ec_frequency.items():
                # 跳过背景EC
                if ec in BACKGROUND_EC:
                    continue

                coverage = ec_weight / total_weight

                if coverage >= 0.8:
                    rule = ComplexFOLRule(
                        rule_type='conditional_necessary',
                        quantifier='forall',
                        head={'necessary_ec': ec},
                        body={'source': src, 'target': tgt},
                        support=total_weight,
                        confidence=coverage,
                        coverage=coverage
                    )
                    rules.append(rule)

        rules.sort(key=lambda r: (r.confidence, r.support), reverse=True)
        return rules

    def get_rules_by_type(self, rule_type: str) -> List[ComplexFOLRule]:
        """按类型获取规则"""
        return [r for r in self.rules if r.rule_type == rule_type]

    def get_rules_for_odor(self, odor: str, role: str = 'any') -> List[ComplexFOLRule]:
        """
        获取与特定气味相关的规则

        Args:
            odor: 气味名称
            role: 'source', 'target', 'any'
        """
        results = []

        for rule in self.rules:
            match = False

            if role in ['source', 'any']:
                if rule.body.get('source') == odor:
                    match = True
                if odor in rule.body.get('sources', []):
                    match = True

            if role in ['target', 'any']:
                if rule.head.get('target') == odor:
                    match = True
                if rule.head.get('excluded_target') == odor:
                    match = True

            if match:
                results.append(rule)

        return results

    def get_rules_for_ec(self, ec: str) -> List[ComplexFOLRule]:
        """获取包含特定EC的规则"""
        results = []

        for rule in self.rules:
            # 检查head
            if rule.head.get('ec') == ec or rule.head.get('necessary_ec') == ec:
                results.append(rule)
                continue

            # 检查body
            if ec in rule.body.get('ec_set', ()):
                results.append(rule)
                continue
            if ec in rule.body.get('ec_sequence', ()):
                results.append(rule)
                continue
            if rule.body.get('ec1') == ec or rule.body.get('ec2') == ec:
                results.append(rule)
                continue

        return results

    def print_rules_summary(self):
        """打印规则摘要"""
        print("\n" + "=" * 80)
        print("规则摘要")
        print("=" * 80)

        type_counts = Counter(r.rule_type for r in self.rules)

        for rule_type, count in type_counts.most_common():
            print(f"  {rule_type}: {count} 条")

        print(f"\n  总计: {len(self.rules)} 条规则")

    def print_top_rules(self, n: int = 10, by_type: bool = True):
        """打印Top规则"""
        print("\n" + "=" * 100)
        print(f"Top {n} 复杂一阶逻辑规则")
        print("=" * 100)

        if by_type:
            type_counts = Counter(r.rule_type for r in self.rules)

            for rule_type in type_counts.keys():
                type_rules = self.get_rules_by_type(rule_type)[:n]

                if not type_rules:
                    continue

                print(f"\n{'─' * 80}")
                print(f"📌 {rule_type.upper()} ({len(self.get_rules_by_type(rule_type))} 条)")
                print(f"{'─' * 80}")

                for i, rule in enumerate(type_rules, 1):
                    print(f"\n  #{i} [support={rule.support:.1f}, conf={rule.confidence:.2f}]")
                    print(f"     {rule.to_fol_string()}")
        else:
            for i, rule in enumerate(self.rules[:n], 1):
                print(f"\n#{i} [{rule.rule_type}] (support={rule.support:.1f}, conf={rule.confidence:.2f})")
                print(f"   {rule.to_fol_string()}")

    def export_rules(self, filepath: str = 'complex_fol_rules.json') -> Dict:
        """导出规则到JSON"""

        # 按类型分组
        rules_by_type = defaultdict(list)
        for rule in self.rules:
            rules_by_type[rule.rule_type].append(rule)

        export_data = {
            'summary': {
                'total_rules': len(self.rules),
                'by_type': {k: len(v) for k, v in rules_by_type.items()},
                'parameters': {
                    'min_support': self.min_support,
                    'min_confidence': self.min_confidence,
                    'high_confidence': self.high_confidence
                }
            },
            'rules': {},
            'prolog_clauses': []
        }

        # 导出每种类型的规则
        for rule_type, type_rules in rules_by_type.items():
            export_data['rules'][rule_type] = [
                {
                    'id': i,
                    'fol_string': r.to_fol_string(),
                    'prolog': r.to_prolog(),
                    'head': r.head,
                    'body': {k: list(v) if isinstance(v, tuple) else v
                             for k, v in r.body.items()},
                    'support': round(r.support, 2),
                    'confidence': round(r.confidence, 3),
                    'coverage': round(r.coverage, 3)
                }
                for i, r in enumerate(type_rules)
            ]

        # 生成Prolog子句
        export_data['prolog_clauses'] = [r.to_prolog() for r in self.rules[:300]]

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 规则已导出到: {filepath}")
        return export_data


# =============================================================================
# PART 4: 更新查询引擎
# =============================================================================

class OdorRuleQueryEngine:
    """气味规则查询引擎 - 支持复杂规则"""

    def __init__(self, pattern_index: Dict, rule_extractor: ComplexRuleExtractor):
        self.index = pattern_index
        self.extractor = rule_extractor

    def query_transformation(self, source_odor: str, top_k: int = 10) -> List[Dict]:
        """任务A: 给定源气味，预测可能的转化"""
        results = []
        pair_to_chains = self.index['pair_to_chains']
        source_to_targets = self.index['source_to_targets']

        if source_odor not in source_to_targets:
            return []

        for target_odor, total_weight in source_to_targets[source_odor].most_common(top_k * 2):
            chains = pair_to_chains.get((source_odor, target_odor), {})
            if not chains:
                continue

            best_chain, chain_weight = chains.most_common(1)[0]
            results.append({
                'source': source_odor,
                'target': target_odor,
                'ec_chain': list(best_chain),
                'score': chain_weight,
                'rule': f"{source_odor} --[{' → '.join(best_chain)}]--> {target_odor}"
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def query_ec_chain(self, source_odor: str, target_odor: str, top_k: int = 5) -> List[Dict]:
        """任务B: 给定气味对，推断EC链"""
        pair_to_chains = self.index['pair_to_chains']
        chains = pair_to_chains.get((source_odor, target_odor), {})

        results = []
        for ec_chain, weight in chains.most_common(top_k):
            results.append({
                'ec_chain': list(ec_chain),
                'score': weight,
                'rule': f"{source_odor} --[{' → '.join(ec_chain)}]--> {target_odor}"
            })
        return results

    def query_necessary_ec(self, source_odor: str, target_odor: str) -> List[str]:
        """查询某转化的必要EC"""
        rules = self.extractor.get_rules_by_type('conditional_necessary')
        necessary_ecs = []

        for rule in rules:
            if (rule.body.get('source') == source_odor and
                    rule.body.get('target') == target_odor):
                necessary_ecs.append(rule.head['necessary_ec'])

        return necessary_ecs

    def query_excluded_targets(self, source_odor: str, ec_sequence: tuple) -> List[str]:
        """查询某EC序列不可能产生的目标气味"""
        rules = self.extractor.get_rules_by_type('exclusion')
        excluded = []

        for rule in rules:
            if (rule.body.get('source') == source_odor and
                    rule.body.get('ec_sequence') == ec_sequence):
                excluded.append(rule.head['excluded_target'])

        return excluded

    def demo_queries(self):
        """演示查询功能"""
        print("\n" + "=" * 80)
        print("规则查询演示")
        print("=" * 80)

        # 任务A
        print("\n📌 任务A: 给定 'mint' 气味，预测可能的转化")
        print("-" * 60)
        results = self.query_transformation('mint', top_k=5)
        for r in results:
            print(f"  {r['rule']}  (score={r['score']:.1f})")

        # 任务B
        print("\n📌 任务B: 给定 'phenolic' → 'woody'，推断EC链")
        print("-" * 60)
        results = self.query_ec_chain('phenolic', 'woody', top_k=5)
        for r in results:
            print(f"  {r['rule']}  (score={r['score']:.1f})")

        # 必要EC查询
        print("\n📌 任务C: 查询 'mint' → 'woody' 的必要EC")
        print("-" * 60)
        necessary = self.query_necessary_ec('mint', 'woody')
        if necessary:
            print(f"  必要EC: {', '.join(necessary)}")
        else:
            print("  未找到必要EC规则")

        # 规则类型查询
        print("\n📌 查询与 'mint' 相关的复杂规则")
        print("-" * 60)
        fol_rules = self.extractor.get_rules_for_odor('mint')
        for r in fol_rules[:5]:
            print(f"  [{r.rule_type}] {r.to_fol_string()}")

def get_ec_category(ec_number):
    """
    EC Function Classification (Biological Significance)
    1.x.x.x - Oxidoreductase
    2.x.x.x - Transferase
    3.x.x.x - Hydrolase
    4.x.x.x - Lyase
    5.x.x.x - Isomerase
    6.x.x.x - Ligase
    """
    categories = {
        '1': 'Oxidoreductase',
        '2': 'Transferase',
        '3': 'Hydrolase',
        '4': 'Lyase',
        '5': 'Isomerase',
        '6': 'Ligase'
    }
    if ec_number and '.' in ec_number:
        return categories.get(ec_number.split('.')[0], 'Unknown')
    return 'Unknown'


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------------------
# Figure 1: Pyramid overview
# ---------------------------
def plot_1_data_overview(data, output_dir):
    """Figure 1: Data Scale Pyramid"""
    fig, ax = plt.subplots(figsize=(10, 6))

    summary = data.get('summary', {})

    stages = ['Pathways', 'Odor Events\n(Cartesian)', 'Unique Triplets', 'Unique EC\nSequences']
    values = [
        summary.get('pathways', 0),
        summary.get('odor_events', 0),
        summary.get('unique_triplets', 0),
        summary.get('unique_ec_sequences', 0)
    ]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(values)))
    bars = ax.barh(stages, values, color=colors, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width + (max(values) * 0.02 if max(values) > 0 else 1),
                bar.get_y() + bar.get_height()/2,
                f'{val:,}', ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Data Scale: From Pathways to Patterns',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, max(values) * 1.15 if max(values) > 0 else 1)

    expansion_ratio = summary.get('expansion_ratio', 0.0)
    textstr = f"Expansion Ratio: {expansion_ratio:.2f}x\n(Avg odors per compound pair)"
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes,
            fontsize=10, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / '01_data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Data Scale Pyramid")


# -----------------------------------------
# Figure 2: Odor category distributions
# -----------------------------------------
def plot_2_odor_distribution(data, output_dir):
    """Figure 2: Odor Category Distribution (Source vs Target)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    source_data = data.get('top_source_odors', [])[:15]
    source_names = [d.get('odor', '') for d in source_data]
    source_weights = [d.get('total_weight', 0.0) for d in source_data]

    ax1.barh(source_names, source_weights, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Weighted Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Top 15 Source Odors', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()

    target_data = data.get('top_target_odors', [])[:15]
    target_names = [d.get('odor', '') for d in target_data]
    target_weights = [d.get('total_weight', 0.0) for d in target_data]

    ax2.barh(target_names, target_weights, color='coral', edgecolor='black')
    ax2.set_xlabel('Weighted Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Top 15 Target Odors', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()

    plt.suptitle('Odor Category Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '02_odor_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Odor Category Distribution")


# -----------------------------------------
# Figure 3: EC sequence length distribution
# -----------------------------------------
def plot_3_pathway_length_distribution(data, output_dir):
    """
    Figure 3: EC Sequence Length Distribution
    Uses as many triplets as available (safe fallback).
    """
    lengths = []
    top_ec_sequences = data.get('top_ec_sequences', [])
    top_triplets = data.get('top_triplets', [])

    # Prefer triplets (each triplet has a concrete ec_sequence)
    if len(top_triplets) > 0:
        # ✅ 从 2000 改成 5000（因为现在有更多数据了）
        sample_size = min(10000, len(top_triplets))  # 原来是 2000
        sampled = random.sample(top_triplets, sample_size) if len(top_triplets) > sample_size else top_triplets
        lengths.extend([len(t.get('ec_sequence', [])) for t in sampled])
        source_tag = f"Sampled {len(sampled):,} triplets"
    elif len(top_ec_sequences) > 0:
        # Fallback: use top_ec_sequences list
        lengths.extend([len(item.get('ec_sequence', [])) for item in top_ec_sequences])
        source_tag = f"From top_ec_sequences ({len(top_ec_sequences)})"
    else:
        # Ultimate fallback: show an empty chart with notice
        lengths = []
        source_tag = "No sequence data available"

    fig, ax = plt.subplots(figsize=(10, 6))

    if lengths:
        counts = Counter(lengths)
        sorted_lengths = sorted(counts.keys())
        freqs = [counts[l] for l in sorted_lengths]
        bars = ax.bar(sorted_lengths, freqs, color='mediumseagreen', edgecolor='black', alpha=0.85)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

        avg_length = np.mean(lengths)
        median_length = np.median(lengths)
        textstr = f'{source_tag}\nMean: {avg_length:.1f}\nMedian: {median_length:.1f}'
        ax.set_xticks(sorted_lengths)
    else:
        textstr = f'{source_tag}'
        ax.bar([0], [0], color='mediumseagreen', edgecolor='black', alpha=0.85)
        ax.set_xticks([])

    ax.set_xlabel('Pathway Length (# of EC steps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('EC Sequence Length Distribution', fontsize=14, fontweight='bold', pad=20)

    ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
            fontsize=11, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_dir / '03_pathway_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: EC Sequence Length Distribution")


# -----------------------------------------
# Figure 4: Source→Target heatmap
# -----------------------------------------
def plot_4_odor_transformation_heatmap(data, output_dir):
    """Figure 4: Odor Transformation Heatmap (Top source × Top target)"""
    top_sources = [d.get('odor', '') for d in data.get('top_source_odors', [])[:15]]
    top_targets = [d.get('odor', '') for d in data.get('top_target_odors', [])[:15]]

    matrix = np.zeros((len(top_sources), len(top_targets)))
    top_triplets = data.get('top_triplets', [])
    # use_n = min(500, len(top_triplets))  # be explicit
    use_n = min(5000, len(top_triplets))  # ✅ 从 500 改成 1000

    for triplet in top_triplets[:use_n]:
        s = triplet.get('source_odor', '')
        t = triplet.get('target_odor', '')
        freq = float(triplet.get('weighted_frequency', 0.0))

        if s in top_sources and t in top_targets:
            i = top_sources.index(s)
            j = top_targets.index(t)
            matrix[i, j] += freq

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

    plt.tight_layout()
    plt.savefig(output_dir / '04_transformation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: Odor Transformation Heatmap")


# -----------------------------------------
# Figure 5: Top EC sequences
# -----------------------------------------
def plot_5_top_ec_sequences(data, output_dir):
    """Figure 5: Top EC Sequences Bar Chart"""
    fig, ax = plt.subplots(figsize=(12, 10))

    top_ecs = data.get('top_ec_sequences', [])[:20]
    ec_labels = [' → '.join(item.get('ec_sequence', [])) for item in top_ecs]
    freqs = [float(item.get('weighted_frequency', 0.0)) for item in top_ecs]

    colors_map = {
        'Oxidoreductase': '#FF6B6B',
        'Transferase':    '#4ECDC4',
        'Hydrolase':      '#45B7D1',
        'Lyase':          '#FFA07A',
        'Isomerase':      '#98D8C8',
        'Ligase':         '#F7DC6F',
        'Unknown':        '#B0B0B0'
    }

    used_cats = []
    colors = []
    for item in top_ecs:
        seq = item.get('ec_sequence', [])
        first_ec = seq[0] if seq else ''
        category = get_ec_category(first_ec)
        colors.append(colors_map.get(category, '#B0B0B0'))
        used_cats.append(category)

    bars = ax.barh(range(len(ec_labels)), freqs, color=colors, edgecolor='black', linewidth=1)

    ax.set_yticks(range(len(ec_labels)))
    ax.set_yticklabels(ec_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Weighted Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 EC Sequences (Odor Space)', fontsize=14, fontweight='bold', pad=20)

    # Legend only for used categories
    unique_used = []
    for c in used_cats:
        if c not in unique_used:
            unique_used.append(c)
    legend_patches = [mpatches.Patch(color=colors_map.get(cat, '#B0B0B0'), label=cat) for cat in unique_used]
    if legend_patches:
        ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / '05_top_ec_sequences.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: Top EC Sequences")


# -----------------------------------------
# Figure 6: Sankey-like flow preview
# -----------------------------------------
def plot_6_sankey_preview(data, output_dir):
    """Figure 6: Odor Transformation Flow (Sankey-like)"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Fallback to top entries if strict threshold filters everything out
    top_triplets_all = data.get('top_triplets', [])
    filtered = [t for t in top_triplets_all if float(t.get('weighted_frequency', 0.0)) > 10]
    top_triplets = (filtered if filtered else top_triplets_all)[:30]
    sources = list({t.get('source_odor', '') for t in top_triplets})[:8]
    targets = list({t.get('target_odor', '') for t in top_triplets})[:8]

    filtered = [t for t in top_triplets if t.get('source_odor', '') in sources and t.get('target_odor', '') in targets]

    y_source = np.linspace(0, 1, len(sources)) if sources else []
    y_target = np.linspace(0, 1, len(targets)) if targets else []

    source_pos = {s: y for s, y in zip(sources, y_source)}
    target_pos = {t: y for t, y in zip(targets, y_target)}

    for triplet in filtered:
        s = triplet.get('source_odor', '')
        t = triplet.get('target_odor', '')
        freq = float(triplet.get('weighted_frequency', 0.0))

        if s in source_pos and t in target_pos:
            y1 = source_pos[s]
            y2 = target_pos[t]

            x = np.linspace(0.2, 0.8, 100)
            y = y1 + (y2 - y1) * (3 * x**2 - 2 * x**3)

            width = 1.0 + 0.3 * np.sqrt(max(freq, 0.0))         # sqrt scaling for readability
            alpha = clamp(freq / 50.0, 0.1, 0.8)                # softer alpha scaling
            ax.plot(x, y, alpha=alpha, linewidth=width, color='steelblue')

    for s, y in source_pos.items():
        ax.text(0.15, y, s, ha='right', va='center', fontsize=10, fontweight='bold')
    for t, y in target_pos.items():
        ax.text(0.85, y, t, ha='left', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    ax.set_title('Major Odor Transformation Flows\n(Line width ∝ sqrt(frequency))',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / '06_transformation_flow.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 6: Transformation Flow")


# -----------------------------------------
# Figure 7a: EC × Source Odor heatmap
# Figure 7b: EC × Target Odor heatmap
# -----------------------------------------
def plot_7_ec_by_odor_category_source(data, output_dir):
    """Figure 7a: EC Sequence usage by Source Odor"""
    top_ecs = data.get('top_ec_sequences', [])[:10]
    top_odors = [d.get('odor', '') for d in data.get('top_source_odors', [])[:15]]

    matrix = np.zeros((len(top_ecs), len(top_odors)))
    top_triplets = data.get('top_triplets', [])[:1000]

    for i, ec_item in enumerate(top_ecs):
        ec_seq = tuple(ec_item.get('ec_sequence', []))
        for triplet in top_triplets:
            if tuple(triplet.get('ec_sequence', [])) == ec_seq:
                s = triplet.get('source_odor', '')
                if s in top_odors:
                    j = top_odors.index(s)
                    matrix[i, j] += float(triplet.get('weighted_frequency', 0.0))

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, cmap='Greens', aspect='auto')

    ec_labels = [' → '.join(item.get('ec_sequence', [])[:3]) + '...' if len(item.get('ec_sequence', [])) > 3
                 else ' → '.join(item.get('ec_sequence', [])) for item in top_ecs]

    ax.set_xticks(np.arange(len(top_odors)))
    ax.set_yticks(np.arange(len(top_ecs)))
    ax.set_xticklabels(top_odors, rotation=45, ha='right')
    ax.set_yticklabels(ec_labels, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weighted Frequency', rotation=270, labelpad=20, fontweight='bold')

    ax.set_title('EC Sequence Usage by Source Odor Category',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Source Odor', fontsize=12, fontweight='bold')
    ax.set_ylabel('EC Sequence', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / '07a_ec_by_source.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 7a: EC × Source Odor")


def plot_7_ec_by_odor_category_target(data, output_dir):
    """Figure 7b: EC Sequence usage by Target Odor"""
    top_ecs = data.get('top_ec_sequences', [])[:10]
    top_odors = [d.get('odor', '') for d in data.get('top_target_odors', [])[:15]]

    matrix = np.zeros((len(top_ecs), len(top_odors)))
    top_triplets = data.get('top_triplets', [])[:1000]

    for i, ec_item in enumerate(top_ecs):
        ec_seq = tuple(ec_item.get('ec_sequence', []))
        for triplet in top_triplets:
            if tuple(triplet.get('ec_sequence', [])) == ec_seq:
                t = triplet.get('target_odor', '')
                if t in top_odors:
                    j = top_odors.index(t)
                    matrix[i, j] += float(triplet.get('weighted_frequency', 0.0))

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, cmap='Purples', aspect='auto')

    ec_labels = [' → '.join(item.get('ec_sequence', [])[:3]) + '...' if len(item.get('ec_sequence', [])) > 3
                 else ' → '.join(item.get('ec_sequence', [])) for item in top_ecs]

    ax.set_xticks(np.arange(len(top_odors)))
    ax.set_yticks(np.arange(len(top_ecs)))
    ax.set_xticklabels(top_odors, rotation=45, ha='right')
    ax.set_yticklabels(ec_labels, fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weighted Frequency', rotation=270, labelpad=20, fontweight='bold')

    ax.set_title('EC Sequence Usage by Target Odor Category',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Target Odor', fontsize=12, fontweight='bold')
    ax.set_ylabel('EC Sequence', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / '07b_ec_by_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 7b: EC × Target Odor")


# -----------------------------------------
# Figure 8a/8b: EC function pies
# -----------------------------------------
def _aggregate_ec_categories(top_ec_sequences, normalized=True):
    """
    Aggregate EC categories.
    normalized=True  -> pathway length normalized (weight divided by #steps)
    normalized=False -> by-step (raw accumulation)
    """
    ec_categories = defaultdict(float)
    for ec_item in top_ec_sequences:
        seq = ec_item.get('ec_sequence', [])
        freq = float(ec_item.get('weighted_frequency', 0.0))
        if not seq:
            continue
        if normalized:
            weight_per_ec = freq / len(seq)
            for ec in seq:
                c = get_ec_category(ec)
                ec_categories[c] += weight_per_ec
        else:
            for ec in seq:
                c = get_ec_category(ec)
                ec_categories[c] += freq
    return ec_categories


def plot_8_ec_function_pies(data, output_dir):
    """Figure 8a: Normalized; Figure 8b: By-step (raw)"""
    top_ecs = data.get('top_ec_sequences', [])[:100]

    # 8a: normalized by pathway length
    ec_norm = _aggregate_ec_categories(top_ecs, normalized=True)
    _plot_pie_from_categories(ec_norm,
                              title='EC Function Distribution (Normalized by Pathway Length)',
                              outpath=output_dir / '08a_ec_function_pie_normalized.png')

    # 8b: by-step (raw accumulation)
    ec_raw = _aggregate_ec_categories(top_ecs, normalized=False)
    _plot_pie_from_categories(ec_raw,
                              title='EC Function Distribution (By Step / Raw Accumulation)',
                              outpath=output_dir / '08b_ec_function_pie_by_step.png')

    print("✓ Figure 8a/8b: EC Function Pies")


def _plot_pie_from_categories(ec_categories, title, outpath):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Keep only positive values
    pairs = [(k, v) for k, v in ec_categories.items() if v > 0]
    if not pairs:
        pairs = [('No Data', 1.0)]
    pairs.sort(key=lambda x: -x[1])

    categories = [k for k, _ in pairs]
    values = [v for _, v in pairs]

    color_map = {
        'Oxidoreductase': '#FF6B6B',
        'Transferase':    '#4ECDC4',
        'Hydrolase':      '#45B7D1',
        'Lyase':          '#FFA07A',
        'Isomerase':      '#98D8C8',
        'Ligase':         '#F7DC6F',
        'Unknown':        '#B0B0B0',
        'No Data':        '#CCCCCC'
    }
    colors = [color_map.get(cat, '#B0B0B0') for cat in categories]

    wedges, texts, autotexts = ax.pie(
        values, labels=categories, autopct='%1.1f%%',
        colors=colors, startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


# -----------------------------------------
# Figure 9: Representative case studies
# -----------------------------------------
def plot_9_case_studies(data, output_dir):
    """Figure 9: Representative Case Studies (fixed save/close placement)"""
    triplets = data.get('top_triplets', [])
    if not triplets:
        print("! Figure 9 skipped (no triplets)")
        return

    # pick up to 3 cases safely
    case_indices = [0, 5, 15]
    case_indices = [i for i in case_indices if i < len(triplets)]
    if not case_indices:
        case_indices = list(range(min(3, len(triplets))))

    fig, axes = plt.subplots(len(case_indices), 1, figsize=(12, 10))
    if len(case_indices) == 1:
        axes = [axes]  # normalize

    for ax_idx, ax in enumerate(axes):
        triplet = triplets[case_indices[ax_idx]]
        s_odor = triplet.get('source_odor', '')
        t_odor = triplet.get('target_odor', '')
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

        ax.text(x_pos[0], 0.5, s_odor, ha='right', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.text(x_pos[-1], 0.5, t_odor, ha='left', va='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

        title = f"Case {ax_idx + 1}: {s_odor} → {t_odor} (freq: {freq:.1f})"
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')

        if examples:
            example_strs = [f"{e.get('src_compound', '')}→{e.get('tgt_compound', '')}" for e in examples[:2]]
            ex_text = f"Examples: {', '.join(example_strs)}"
            ax.text(0.5, 0.1, ex_text, ha='center', va='top',
                    fontsize=9, style='italic', transform=ax.transAxes)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.suptitle('Representative Odor Transformation Pathways',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / '09_case_studies.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 9: Case Studies")


# -----------------------------------------
# Figure 10: Key findings dashboard
# -----------------------------------------
def plot_10_key_findings(data, output_dir):
    """Figure 10: Key Findings Summary"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # 1) Top 3 transformations
    ax1 = fig.add_subplot(gs[0, 0])
    top3 = data.get('top_triplets', [])[:3]
    labels = [f"{t.get('source_odor','')}→{t.get('target_odor','')}" for t in top3]
    values = [float(t.get('weighted_frequency', 0.0)) for t in top3]
    ax1.barh(labels, values, color=['gold', 'silver', '#CD7F32'], edgecolor='black')
    ax1.set_xlabel('Frequency', fontweight='bold')
    ax1.set_title('[Top 3] Transformations', fontweight='bold')
    ax1.invert_yaxis()

    # 2) Most common EC sequence
    ax2 = fig.add_subplot(gs[0, 1])
    top_ec = data.get('top_ec_sequences', [{}])[0]
    ec_text = ' → '.join(top_ec.get('ec_sequence', []))
    freq_text = f"Frequency: {float(top_ec.get('weighted_frequency', 0.0)):.0f}"
    ax2.text(0.5, 0.6, ec_text if ec_text else 'N/A', ha='center', va='center',
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax2.text(0.5, 0.3, freq_text, ha='center', va='center', fontsize=10)
    ax2.set_title('[Most Common] EC Sequence', fontweight='bold')
    ax2.axis('off')

    # 3) Coverage statistics
    ax3 = fig.add_subplot(gs[1, 0])
    summary = data.get('summary', {})
    stats_text = f"""
    Total Pathways: {summary.get('pathways',0):,}
    Unique Odor Pairs: {summary.get('unique_triplets',0):,}
    Source Odors: {summary.get('unique_source_odors',0)}
    Target Odors: {summary.get('unique_target_odors',0)}
    EC Sequences: {summary.get('unique_ec_sequences',0):,}
    """
    ax3.text(0.1, 0.5, stats_text, ha='left', va='center',
             fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax3.set_title('[Statistics] Coverage Statistics', fontweight='bold')
    ax3.axis('off')

    # 4) Biological insights (static notes)
    ax4 = fig.add_subplot(gs[1, 1])
    bio_text = """
    Key Insights:
    • Oxidoreductases (EC 1.x) dominate
    • 2-3 step pathways most common
    • Phenolic→Woody appears prominently
    • Mint/Floral act as hubs
    """
    ax4.text(0.1, 0.5, bio_text, ha='left', va='center',
             fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax4.set_title('[Biology] Key Insights', fontweight='bold')
    ax4.axis('off')

    # 5) Notable specific transformations (lower-frequency)
    ax5 = fig.add_subplot(gs[2, :])
    triplets = data.get('top_triplets', [])
    rare_triplets = [t for t in triplets[20:50] if float(t.get('weighted_frequency', 0.0)) < 15][:6]

    if rare_triplets:
        labels = [f"{t.get('source_odor','')}→{t.get('target_odor','')}\n({' → '.join(t.get('ec_sequence', [])[:2])}...)"
                  for t in rare_triplets]
        values = [float(t.get('weighted_frequency', 0.0)) for t in rare_triplets]
        ax5.bar(range(len(labels)), values, color='mediumpurple', edgecolor='black', alpha=0.7)
        ax5.set_xticks(range(len(labels)))
        ax5.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('[Notable] Specific Transformations (Lower frequency but interesting)',
                      fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No notable low-frequency cases found.',
                 ha='center', va='center')
        ax5.axis('off')

    plt.suptitle('Key Findings Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_dir / '10_key_findings.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 10: Key Findings")


def plot_11_empirical_dashboard(output_dir):
    """
    Figure 11: Paper-ready composite dashboard (2x3 panels).
    Combines key V5 outputs into one large summary figure.
    """
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
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"({tag}) {title}", fontsize=12, fontweight='bold', pad=8)

    fig.suptitle(
        "Empirical Overview of Odor Transformation Rules",
        fontsize=18,
        fontweight='bold',
        y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_dir / '11_empirical_dashboard.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / '11_empirical_dashboard.pdf', bbox_inches='tight')
    plt.close(fig)
    print("✓ Figure 11: Empirical Dashboard (PNG + PDF)")

# =============================================================================
# PART 4: 数据导出 - 连接 pattern_index → 可视化格式
# =============================================================================

def export_for_visualization(
        pattern_index: Dict,
        pathways: List[Dict],
        events: List[Dict],
        output_json: str = "odor_level_patterns_weighted.json"
) -> Dict:
    """
    将 build_odor_pattern_index() 的输出转换为可视化脚本期望的 JSON 格式

    Args:
        pattern_index: build_odor_pattern_index() 的输出
        pathways: analyzer.find_odor_pathways() 的输出
        events: expand_to_odor_events() 的输出
        output_json: 输出文件路径

    Returns:
        可视化数据字典
    """
    triplet_counts = pattern_index['triplet_counts']
    triplet_examples = pattern_index['triplet_examples']
    ec_seq_counts = pattern_index['ec_seq_counts']
    source_to_targets = pattern_index['source_to_targets']

    # 1. 构建 top_triplets
    top_triplets = []
    for (src, ec_seq, tgt), weight in sorted(triplet_counts.items(), key=lambda x: -x[1])[:10000]:
        examples = triplet_examples.get((src, ec_seq, tgt), [])
        top_triplets.append({
            'source_odor': src,
            'target_odor': tgt,
            'ec_sequence': list(ec_seq),
            'weighted_frequency': round(weight, 2),
            'examples': [
                {'src_compound': ex['src_cpd'], 'tgt_compound': ex['tgt_cpd']}
                for ex in examples
            ]
        })

    # 2. 构建 top_ec_sequences
    top_ec_sequences = []
    for ec_seq, weight in sorted(ec_seq_counts.items(), key=lambda x: -x[1])[:500]:
        top_ec_sequences.append({
            'ec_sequence': list(ec_seq),
            'weighted_frequency': round(weight, 2)
        })

    # 3. 构建 top_source_odors
    source_totals = Counter()
    for src, targets in source_to_targets.items():
        source_totals[src] = sum(targets.values())

    top_source_odors = [
        {'odor': odor, 'total_weight': round(weight, 2)}
        for odor, weight in source_totals.most_common(50)
    ]

    # 4. 构建 top_target_odors
    target_totals = Counter()
    for (src, ec_seq, tgt), weight in triplet_counts.items():
        target_totals[tgt] += weight

    top_target_odors = [
        {'odor': odor, 'total_weight': round(weight, 2)}
        for odor, weight in target_totals.most_common(50)
    ]

    # 5. 构建 summary
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

    # 6. 组装最终数据
    viz_data = {
        'summary': summary,
        'top_triplets': top_triplets,
        'top_ec_sequences': top_ec_sequences,
        'top_source_odors': top_source_odors,
        'top_target_odors': top_target_odors
    }

    # 7. 保存 JSON
    with open(output_json, 'w') as f:
        json.dump(viz_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 可视化数据已导出: {output_json}")
    print(f"   - {len(top_triplets)} triplets")
    print(f"   - {len(top_ec_sequences)} EC sequences")

    return viz_data



# =============================================================================
# PART 7: 交互式网络可视化
# =============================================================================

def visualize_odor_network(analyzer, output_file="odor_network.html",
                           max_nodes=500, only_odorous=False):
    """
    交互式网络可视化

    Args:
        analyzer: OdorPathwayAnalyzer 实例
        output_file: 输出HTML文件名
        max_nodes: 最大节点数量
        only_odorous: 是否只显示气味化合物相关节点
    """
    from pyvis.network import Network

    G = analyzer.reaction_graph

    # 筛选子图：只保留气味化合物及其直接连接的反应
    if only_odorous:
        odor_nodes = [n for n, d in G.nodes(data=True)
                      if d.get('node_type') == 'odor_compound']

        # 获取与气味化合物直接相连的反应
        related_reactions = set()
        for node in odor_nodes:
            related_reactions.update(G.predecessors(node))
            related_reactions.update(G.successors(node))

        subgraph_nodes = set(odor_nodes) | related_reactions
        G_sub = G.subgraph(subgraph_nodes).copy()
    else:
        G_sub = G

    # 限制节点数量（按度数排序保留高连接度节点）
    if G_sub.number_of_nodes() > max_nodes:
        degrees = dict(G_sub.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G_sub = G_sub.subgraph(top_nodes).copy()

    print(f"  可视化子图: {G_sub.number_of_nodes()} 节点, {G_sub.number_of_edges()} 边")

    # 创建Pyvis网络
    net = Network(height="900px", width="100%",
                  bgcolor="#222222", font_color="white",
                  directed=True)

    # 颜色方案
    colors = {
        'odor_compound': '#FF6B6B',  # 红色 - 气味化合物
        'intermediate': '#4ECDC4',  # 青色 - 中间体
        'reaction': '#FFE66D'  # 黄色 - 反应
    }

    # 统计
    node_counts = {'odor_compound': 0, 'intermediate': 0, 'reaction': 0}

    # 添加节点
    for node, data in G_sub.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        node_counts[node_type] = node_counts.get(node_type, 0) + 1

        if node_type == 'odor_compound':
            dominant_odors = data.get('dominant_odors', [])
            label = node
            size = 25
            title = f"<b>{node}</b><br>Type: Odor Compound<br>Odors: {', '.join(dominant_odors)}"
        elif node_type == 'reaction':
            ec_nums = data.get('ec_numbers', [])
            label = node.replace('R_', '')
            size = 12
            ec_str = ', '.join(ec_nums[:3]) if ec_nums else 'No EC'
            title = f"<b>{node}</b><br>Type: Reaction<br>EC: {ec_str}"
        else:
            label = node
            size = 10
            title = f"<b>{node}</b><br>Type: Intermediate"

        net.add_node(node,
                     label=label,
                     color=colors.get(node_type, '#888888'),
                     size=size,
                     title=title,
                     font={'size': 10})

    # 添加边
    for u, v in G_sub.edges():
        net.add_edge(u, v, arrows='to', color='#555555', width=0.5)

    # 物理引擎设置 - 优化大图显示
    net.set_options('''
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {"size": 10}
      },
      "edges": {
        "color": {"inherit": false},
        "smooth": {"type": "continuous"}
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "centralGravity": 0.01,
          "springLength": 150,
          "springConstant": 0.02,
          "damping": 0.4
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 200,
          "updateInterval": 25
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    ''')

    net.save_graph(output_file)

    print(f"  节点组成: 气味化合物={node_counts.get('odor_compound', 0)}, "
          f"反应={node_counts.get('reaction', 0)}, "
          f"中间体={node_counts.get('intermediate', 0)}")
    print(f"✅ 交互式网络已保存到: {output_file}")
    print(f"   用浏览器打开该文件即可查看")

    return net

# =============================================================================
# PART 6: 主函数 (整合版)
# =============================================================================

def main():
    """
    整合版主函数：数据生成 → 笛卡尔展开 → 规则提取 → 可视化
    """
    print("=" * 80)
    print("气味规则提取与可视化系统 (整合版)")
    print("=" * 80)

    # =========================================================================
    # Step 1: 数据生成
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: 数据生成")
    print("=" * 80)

    analyzer = OdorPathwayAnalyzer()

    if not analyzer.load_tgsc_data("../02_kegg_mapping/tgsc_to_kegg.csv"):
        print("❌ 无法加载TGSC数据")
        return

    analyzer.download_kegg_data()

    if not analyzer.build_odor_network(max_depth=2, use_main_pairs=True):
        print("❌ 无法构建网络")
        return

    pathways = analyzer.find_odor_pathways(max_length=5, max_paths_per_pair=2, use_cache=True)

    if not pathways:
        print("❌ 未找到pathways")
        return

    print(f"\n✅ 获得 {len(pathways):,} 条 pathways")

    # =========================================================================
    # Step 2: 笛卡尔展开
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: 笛卡尔展开")
    print("=" * 80)

    events = expand_to_odor_events(pathways)
    print(f"展开为 {len(events):,} 个气味级事件")

    pattern_index = build_odor_pattern_index(events)
    print(f"唯一三元组: {len(pattern_index['triplet_counts']):,}")
    print(f"唯一EC序列: {len(pattern_index['ec_seq_counts']):,}")

    # =========================================================================
    # Step 3: 规则提取
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: 复杂一阶逻辑规则提取")
    print("=" * 80)

    extractor = ComplexRuleExtractor(
        pattern_index,
        min_support=2.0,
        min_confidence=0.1,
        high_confidence=0.8,
        exclusion_threshold=0.95
    )

    rules = extractor.extract_all_rules()
    extractor.print_rules_summary()
    extractor.print_top_rules(n=5, by_type=True)
    extractor.export_rules('complex_fol_rules.json')

    # =========================================================================
    # Step 4: 导出可视化数据
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: 导出可视化数据")
    print("=" * 80)

    viz_data = export_for_visualization(
        pattern_index, pathways, events,
        output_json="odor_level_patterns_weighted.json"
    )

    # =========================================================================
    # Step 5: 生成可视化图表
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: 生成可视化图表")
    print("=" * 80)

    output_dir = Path("./visualization_charts")
    output_dir.mkdir(exist_ok=True)

    # 调用所有可视化函数
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



    # =========================================================================
    # Step 6: 查询演示
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: 规则查询演示")
    print("=" * 80)

    query_engine = OdorRuleQueryEngine(pattern_index, extractor)
    query_engine.demo_queries()

    # =========================================================================
    # Step 7: 交互式网络可视化
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: 交互式网络可视化")
    print("=" * 80)

    try:
        visualize_odor_network(
            analyzer,
            output_file="odor_network_interactive.html",
            max_nodes=2000,  # 可调整节点数量
            only_odorous=False  # 只显示气味相关节点
        )
    except ImportError:
        print("❌ 请先安装 pyvis: pip install pyvis")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

    print("\n" + "=" * 80)
    print("🎉 全部完成!")
    print("=" * 80)



if __name__ == "__main__":
    main()
