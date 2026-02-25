"""
v5_data.py — OdorPathwayAnalyzer: KEGG data loading, network building, pathway discovery
"""

import re
import json
import time
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from v5_config import UBIQUITOUS_COMPOUNDS, BACKGROUND_EC, to_ec_level


class OdorPathwayAnalyzer:
    """Optimized analyzer with pathway caching for debugging"""

    @staticmethod
    def parse_kegg_flat(text: str) -> dict:
        """Parse KEGG flat-format text, handling continuation lines."""
        fields = {}
        current_key = None
        for raw_line in text.splitlines():
            key = raw_line[:12].strip()
            val = raw_line[12:].strip()
            if key:
                current_key = key
                if current_key not in fields:
                    fields[current_key] = []
                if val:
                    fields[current_key].append(val)
            elif current_key and val:
                fields[current_key].append(val)
        return {k: " ".join(v) for k, v in fields.items()}

    def __init__(self, cache_dir="../kegg_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.compound_to_reactions = defaultdict(set)
        self.reaction_to_compounds = defaultdict(set)
        self.reaction_reversibility = {}
        self.ec_data = {}

        self.main_pairs_raw = {}
        self.main_pairs_index = {}

        self.tgsc_data = None
        self.odor_attributes = []
        self.compound_profiles = {}
        self.odorous_compounds = set()

        self.reaction_graph = None
        self.pathways = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_tgsc_data(self, csv_file="../02_kegg_mapping/tgsc_to_kegg.csv"):
        """Load TGSC data and identify odorous compounds only"""
        print(f"Loading TGSC-KEGG data from {csv_file}...")
        try:
            self.tgsc_data = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(self.tgsc_data)} TGSC compounds")

            core_cols = {'TGSC ID', 'KEGG_IDs', 'CID', 'IsomericSMILES', 'IUPACName',
                         'Updated_Desc_v2', 'Solvent'}
            self.odor_attributes = []
            for col in self.tgsc_data.columns:
                if col not in core_cols:
                    unique_vals = self.tgsc_data[col].dropna().unique()
                    if len(unique_vals) <= 2 and all(v in [0, 1, 0.0, 1.0] for v in unique_vals):
                        self.odor_attributes.append(col)
            print(f"✓ Identified {len(self.odor_attributes)} odor attributes")

            valid_kegg = self.tgsc_data['KEGG_IDs'].notna() & self.tgsc_data['KEGG_IDs'].str.match(r'^C\d{5}$')
            valid_data = self.tgsc_data[valid_kegg].copy()

            odorless_count = 0
            odorous_count = 0
            for _, row in valid_data.iterrows():
                kegg_id = row['KEGG_IDs']
                odor_profile = row[self.odor_attributes].fillna(0).values.astype(np.int8)
                non_odorless_attributes = [attr for attr in self.odor_attributes if 'odorless' not in attr.lower()]
                odor_indices = [self.odor_attributes.index(attr) for attr in non_odorless_attributes]
                has_odor = any(odor_profile[i] == 1 for i in odor_indices)
                if has_odor:
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

    # ------------------------------------------------------------------
    # KEGG download
    # ------------------------------------------------------------------
    def download_kegg_data(self):
        """Download KEGG reaction data"""
        import requests
        print("Downloading KEGG reaction network...")

        comp_react_file = self.cache_dir / "compound_reaction_links.tsv"
        if not comp_react_file.exists():
            print("  Fetching compound->reaction links...")
            url = "https://rest.kegg.jp/link/reaction/compound"
            response = requests.get(url)
            if response.status_code == 200:
                with open(comp_react_file, 'w') as f:
                    f.write(response.text)

        react_comp_file = self.cache_dir / "reaction_compound_links.tsv"
        if not react_comp_file.exists():
            print("  Fetching reaction->compound links...")
            url = "https://rest.kegg.jp/link/compound/reaction"
            response = requests.get(url)
            if response.status_code == 200:
                with open(react_comp_file, 'w') as f:
                    f.write(response.text)

        self._parse_kegg_links(comp_react_file, react_comp_file)

    def _parse_kegg_links(self, comp_react_file, react_comp_file):
        with open(comp_react_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    compound = parts[0].replace('cpd:', '')
                    reaction = parts[1].replace('rn:', '')
                    if compound in UBIQUITOUS_COMPOUNDS:
                        continue
                    if compound in self.odorous_compounds:
                        self.compound_to_reactions[compound].add(reaction)

        with open(react_comp_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    reaction = parts[0].replace('rn:', '')
                    compound = parts[1].replace('cpd:', '')
                    if compound in UBIQUITOUS_COMPOUNDS:
                        continue
                    self.reaction_to_compounds[reaction].add(compound)

        print(f"✓ Parsed {len(self.compound_to_reactions)} odorous compounds "
              f"and {len(self.reaction_to_compounds)} reactions")
        print(f"  (已过滤 {len(UBIQUITOUS_COMPOUNDS)} 种载体分子)")

    def download_expanded_reaction_metadata(self, expanded_reactions):
        """Download metadata with robust caching"""
        import requests
        print("Downloading metadata with robust caching...")
        print(f"  Requested reactions: {len(expanded_reactions)}")

        cache_file = self.cache_dir / "complete_reaction_cache.json"
        cache_data = {
            'successful': {'ec_data': {}, 'reversibility': {}},
            'failed': {'reactions': set(), 'last_attempt': {}, 'attempt_count': {}},
            'metadata': {'last_updated': None, 'total_attempts': 0, 'success_rate': 0.0}
        }

        if cache_file.exists():
            print(f"  Loading cache: {cache_file}")
            try:
                with open(cache_file, 'r') as f:
                    enhanced_data = json.load(f)
                if 'successful' in enhanced_data:
                    enhanced_data['failed']['reactions'] = set(enhanced_data['failed']['reactions'])
                    cache_data['successful']['ec_data'].update(
                        enhanced_data.get('successful', {}).get('ec_data', {}))
                    cache_data['successful']['reversibility'].update(
                        enhanced_data.get('successful', {}).get('reversibility', {}))
                    cache_data['failed'] = enhanced_data.get('failed', cache_data['failed'])
                    cache_data['metadata'] = enhanced_data.get('metadata', cache_data['metadata'])
                else:
                    for reaction_id, ec_numbers in enhanced_data.items():
                        if isinstance(ec_numbers, list):
                            cache_data['successful']['ec_data'][reaction_id] = ec_numbers
                            cache_data['successful']['reversibility'][reaction_id] = True
                successful_count = len(cache_data['successful']['ec_data'])
                failed_count = len(cache_data['failed']['reactions'])
                print(f"    ✅ Cache loaded: {successful_count} successful, {failed_count} failed")
            except Exception as e:
                print(f"    ❌ Error loading cache: {e}")

        self.ec_data = cache_data['successful']['ec_data'].copy()
        self.reaction_reversibility = cache_data['successful']['reversibility'].copy()

        cached_successful = set(cache_data['successful']['ec_data'].keys())
        previously_failed = cache_data['failed']['reactions']
        never_attempted = expanded_reactions - cached_successful - previously_failed
        need_retry = previously_failed & expanded_reactions

        print(f"  📊 Download analysis:")
        print(f"    ✅ Already successful: {len(cached_successful & expanded_reactions)}")
        print(f"    🆕 Never attempted: {len(never_attempted)}")
        print(f"    🔄 Need retry: {len(need_retry)}")

        if not never_attempted and not need_retry:
            print(f"  🎉 All requested reactions already cached successfully!")
            return

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
                    if 'ENTRY' in text and reaction_id in text:
                        fields = self.parse_kegg_flat(text)
                        ec_numbers = []
                        enzyme_str = fields.get("ENZYME", "")
                        if enzyme_str:
                            ec_numbers = re.findall(r'\d+(?:\.\d+|\.-){1,3}', enzyme_str)
                        if not ec_numbers:
                            orthology_str = fields.get("ORTHOLOGY", "")
                            if orthology_str:
                                ec_matches = re.findall(r'\[EC:([\d\.\-\s]+)\]', orthology_str)
                                for match in ec_matches:
                                    ecs = re.findall(r'\d+(?:\.\d+|\.-){1,3}', match)
                                    ec_numbers.extend(ecs)
                        if ec_numbers:
                            ec_numbers = list(dict.fromkeys(ec_numbers))

                        equation_str = fields.get("EQUATION", "")
                        is_reversible = any(sym in equation_str for sym in ["⇄", "<=>", "<==>", "<->"])
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
                time.sleep(0.35)
            except Exception:
                new_failures.add(reaction_id)
                download_stats['failures'] += 1
                time.sleep(0.5)

        cache_data['successful']['ec_data'].update(new_successful)
        cache_data['successful']['reversibility'].update(new_reversibility)
        cache_data['failed']['reactions'].update(new_failures)
        current_time = time.time()
        for reaction in new_failures:
            cache_data['failed']['last_attempt'][reaction] = current_time
            cache_data['failed']['attempt_count'][reaction] = \
                cache_data['failed']['attempt_count'].get(reaction, 0) + 1

        self.ec_data.update(new_successful)
        self.reaction_reversibility.update(new_reversibility)
        self._save_enhanced_cache(cache_data, cache_file)

        success_rate = download_stats['successes'] / download_stats['attempts'] if download_stats['attempts'] > 0 else 0
        print(f"  📊 Download Results: {download_stats['successes']} successful, "
              f"{download_stats['failures']} failed ({success_rate:.1%})")

    def _save_enhanced_cache(self, cache_data, cache_file):
        try:
            save_copy = cache_data.copy()
            save_copy['failed'] = dict(cache_data['failed'])
            save_copy['failed']['reactions'] = list(cache_data['failed']['reactions'])
            with open(cache_file, 'w') as f:
                json.dump(save_copy, f, indent=2)
            total_successful = len(cache_data['successful']['ec_data'])
            print(f"  💾 Cache saved: {total_successful} reactions, {cache_file.stat().st_size:,} bytes")
        except Exception as e:
            print(f"  💥 Error saving cache: {e}")

    # ------------------------------------------------------------------
    # Main-pair loading
    # ------------------------------------------------------------------
    def load_main_pairs(self, main_pairs_file=None):
        if main_pairs_file is None:
            main_pairs_file = self.cache_dir / "rclass_verification" / "main_pairs.json"
        main_pairs_path = Path(main_pairs_file)
        if not main_pairs_path.exists():
            print(f"⚠️  Main-pair 文件不存在: {main_pairs_path}")
            return False

        print(f"📦 加载 main-pair 数据...")
        try:
            with open(main_pairs_path, 'r') as f:
                self.main_pairs_raw = json.load(f)
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False

        self.main_pairs_index = defaultdict(lambda: defaultdict(set))
        filtered_pairs_count = 0
        for reaction, pairs in self.main_pairs_raw.items():
            for pair in pairs:
                if isinstance(pair, list) and len(pair) == 2:
                    c1, c2 = pair
                else:
                    continue
                if c1 in UBIQUITOUS_COMPOUNDS or c2 in UBIQUITOUS_COMPOUNDS:
                    filtered_pairs_count += 1
                    continue
                self.main_pairs_index[reaction][c1].add(c2)
                if self.reaction_reversibility.get(reaction, True):
                    self.main_pairs_index[reaction][c2].add(c1)

        total_pairs = sum(len(pairs) for pairs in self.main_pairs_raw.values())
        print(f"✅ 成功加载 {len(self.main_pairs_index)} 个反应的 main-pair")
        print(f"   总计 {total_pairs} 个化合物对, 已过滤 {filtered_pairs_count} 个载体分子pair")
        return True

    # ------------------------------------------------------------------
    # Network building
    # ------------------------------------------------------------------
    def build_odor_network(self, max_depth=3, use_main_pairs=False):
        """Build network with ONLY odorous compounds"""
        print(f"Building odor-focused network (depth={max_depth}, use_main_pairs={use_main_pairs})...")

        compounds_to_include = self.odorous_compounds.copy() - UBIQUITOUS_COMPOUNDS
        all_reactions_needed = set()
        print(f"  🎯 Starting with {len(compounds_to_include)} odorous compounds")

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
                    new_compounds.update(self.reaction_to_compounds[reaction] - UBIQUITOUS_COMPOUNDS)

            old_size = len(compounds_to_include)
            compounds_to_include.update(new_compounds)
            print(f"    Added {len(new_compounds)} compounds, total: {len(compounds_to_include)}")
            if len(compounds_to_include) == old_size:
                break

        self.download_expanded_reaction_metadata(all_reactions_needed)

        if use_main_pairs:
            self.load_main_pairs()
            print("  🔗 Using main-pair filtering for edge creation")
        else:
            print("  🔗 Using full reaction connectivity (no main-pair filtering)")

        self.reaction_graph = nx.MultiDiGraph()

        for comp in compounds_to_include:
            if comp in UBIQUITOUS_COMPOUNDS:
                continue
            if comp in self.odorous_compounds:
                profile = self.compound_profiles[comp]
                self.reaction_graph.add_node(comp, node_type='odor_compound',
                                             name=profile['name'],
                                             odor_profile=profile['odor_profile'],
                                             dominant_odors=profile['dominant_odors'])
            else:
                self.reaction_graph.add_node(comp, node_type='intermediate',
                                             name=f"Intermediate_{comp}")

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

            self.reaction_graph.add_node(reaction_node, node_type='reaction', ec_numbers=ec_numbers)
            all_compounds = (self.reaction_to_compounds[reaction] & compounds_to_include) - UBIQUITOUS_COMPOUNDS
            edges_added_for_reaction = 0

            if use_main_pairs and reaction in self.main_pairs_index:
                reactions_with_mainpair += 1
                for c_in, c_out_set in self.main_pairs_index[reaction].items():
                    if c_in not in all_compounds:
                        continue
                    if not self.reaction_graph.has_edge(c_in, reaction_node):
                        self.reaction_graph.add_edge(c_in, reaction_node)
                        edges_added_for_reaction += 1
                    for c_out in c_out_set:
                        if c_out == c_in or c_out not in all_compounds:
                            continue
                        if not self.reaction_graph.has_edge(reaction_node, c_out):
                            self.reaction_graph.add_edge(reaction_node, c_out)
                            edges_added_for_reaction += 1
            elif not use_main_pairs:
                reactions_without_mainpair += 1
                for comp in all_compounds:
                    if not self.reaction_graph.has_edge(comp, reaction_node):
                        self.reaction_graph.add_edge(comp, reaction_node)
                        edges_added_for_reaction += 1
                    if not self.reaction_graph.has_edge(reaction_node, comp):
                        self.reaction_graph.add_edge(reaction_node, comp)
                        edges_added_for_reaction += 1
            else:
                reactions_without_mainpair += 1

            total_edges_added += edges_added_for_reaction

        print(f"✓ Built network: {self.reaction_graph.number_of_nodes()} nodes, "
              f"{self.reaction_graph.number_of_edges()} edges")
        if use_main_pairs:
            coverage = reactions_with_mainpair / len(all_reactions_needed) * 100 if all_reactions_needed else 0
            print(f"  📊 Main-pair: {reactions_with_mainpair} w/ mainpair, "
                  f"{reactions_without_mainpair} w/o, coverage {coverage:.1f}%")
        print(f"  Reactions with EC: {reactions_with_ec}, without: {reactions_without_ec}")

        odor_nodes = sum(1 for n in self.reaction_graph.nodes()
                         if self.reaction_graph.nodes[n].get('node_type') == 'odor_compound')
        intermediate_nodes = sum(1 for n in self.reaction_graph.nodes()
                                 if self.reaction_graph.nodes[n].get('node_type') == 'intermediate')
        reaction_nodes = sum(1 for n in self.reaction_graph.nodes() if n.startswith('R_'))
        print(f"  Odor compounds: {odor_nodes}, Intermediates: {intermediate_nodes}, Reactions: {reaction_nodes}")
        return True

    # ------------------------------------------------------------------
    # Pathway finding
    # ------------------------------------------------------------------
    def _save_pathways_cache(self, pathways, max_length, max_paths_per_pair):
        try:
            cache_file = self.cache_dir / "pathways_cache.pkl"
            cache_data = {
                'pathways': pathways,
                'parameters': {'max_length': max_length, 'max_paths_per_pair': max_paths_per_pair},
                'metadata': {'created_time': time.time(), 'pathway_count': len(pathways),
                             'odor_compounds_count': len(self.odorous_compounds)}
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"  💾 Pathways cached: {len(pathways)} pathways, {cache_file.stat().st_size:,} bytes")
            return True
        except Exception as e:
            print(f"  ❌ Failed to cache pathways: {e}")
            return False

    def _load_pathways_cache(self, max_length, max_paths_per_pair):
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
            print(f"  ✅ Cache loaded: {len(pathways)} pathways, "
                  f"created {time.ctime(metadata.get('created_time', 0))}")
            return pathways
        except Exception as e:
            print(f"  ❌ Failed to load pathway cache: {e}")
            return None

    def find_odor_pathways(self, max_length=5, max_paths_per_pair=1, use_cache=True, force_recompute=False):
        """Enhanced pathway finding with caching support"""
        print(f"Finding pathways ≤{max_length} reaction steps, max {max_paths_per_pair} paths per pair")

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
        print(f"  Total compounds: {len(odor_compounds)}, possible pairs: {total_possible_pairs:,}")

        print("  🔄 Building optimized graph with reaction step weights...")
        simple_graph = self._create_simple_graph_for_pathfinding()

        print("  📊 Pre-computing reachable pairs with single-source Dijkstra...")
        start_time = time.time()
        reachable_pairs = set()
        for src in tqdm(odor_compounds, desc="Computing distances"):
            try:
                distances = nx.single_source_dijkstra_path_length(simple_graph, src, cutoff=max_length)
                for tgt, dist in distances.items():
                    if tgt in self.odorous_compounds and src < tgt and dist <= max_length:
                        reachable_pairs.add((src, tgt))
            except Exception:
                continue

        reachable_pairs = sorted(reachable_pairs)
        print(f"  ✓ Pre-computation done in {time.time() - start_time:.1f}s")
        print(f"  ✓ Found {len(reachable_pairs):,} reachable pairs (≤{max_length} steps)")

        pathways = []
        pairs_with_paths = 0
        total_paths_found = 0
        for source, target in tqdm(reachable_pairs, desc="Yen k-shortest"):
            try:
                pair_pathways = self._find_multiple_paths_optimized(
                    source, target, max_length, max_paths_per_pair, simple_graph)
                if pair_pathways:
                    pairs_with_paths += 1
                    total_paths_found += len(pair_pathways)
                    pathways.extend(pair_pathways)
            except Exception:
                continue

        self.pathways = pathways
        if pathways:
            self._save_pathways_cache(pathways, max_length, max_paths_per_pair)

        connectivity_rate = pairs_with_paths / len(reachable_pairs) * 100 if reachable_pairs else 0
        avg_paths = total_paths_found / pairs_with_paths if pairs_with_paths > 0 else 0
        print(f"\n✓ PATHWAY DISCOVERY COMPLETE: {len(pathways):,} pathways, "
              f"{pairs_with_paths:,}/{len(reachable_pairs):,} connected ({connectivity_rate:.1f}%), "
              f"avg {avg_paths:.1f} paths/pair")
        return pathways

    def _create_simple_graph_for_pathfinding(self):
        simple_graph = nx.DiGraph()
        for node, data in self.reaction_graph.nodes(data=True):
            simple_graph.add_node(node, **data)
        for u, v in self.reaction_graph.edges():
            if u.startswith('R_') == v.startswith('R_'):
                continue
            if not simple_graph.has_edge(u, v):
                simple_graph.add_edge(u, v, weight=0.5)
        return simple_graph

    def _find_multiple_paths_optimized(self, source, target, max_length, max_paths_per_pair, simple_graph):
        found_paths = []
        try:
            from networkx.algorithms.simple_paths import shortest_simple_paths
            path_generator = shortest_simple_paths(simple_graph, source, target, weight='weight')
            paths_collected = 0
            for path in path_generator:
                reaction_steps = sum(1 for node in path if node.startswith('R_'))
                if reaction_steps <= max_length:
                    pathway_info = self._parse_pathway(path, source, target)
                    if pathway_info:
                        pathway_info['path_type'] = 'yen_shortest' if paths_collected == 0 else 'yen_alternative'
                        pathway_info['path_rank'] = paths_collected + 1
                        found_paths.append(pathway_info)
                        paths_collected += 1
                        if paths_collected >= max_paths_per_pair:
                            break
                else:
                    break
        except Exception:
            try:
                path = nx.shortest_path(simple_graph, source, target, weight='weight')
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
        if not path or path[0].startswith('R_'):
            return None
        for i in range(len(path) - 1):
            if path[i].startswith('R_') == path[i + 1].startswith('R_'):
                return None

        pathway_steps = []
        ec_sequence = []
        for i in range(0, len(path) - 1, 2):
            if i + 2 < len(path):
                comp1, reaction, comp2 = path[i], path[i + 1], path[i + 2]
                reaction_info = self.reaction_graph.nodes[reaction]
                ec_numbers = reaction_info.get('ec_numbers', [])
                if ec_numbers:
                    ec3 = to_ec_level(ec_numbers[0], level=3)
                    ec_sequence.append(ec3 if ec3 else 'unknown')
                else:
                    ec_sequence.append('unknown')
                pathway_steps.append({
                    'from': comp1, 'reaction': reaction.replace('R_', ''),
                    'to': comp2, 'ec_numbers': ec_numbers,
                    'has_ec_data': len(ec_numbers) > 0
                })

        if not pathway_steps:
            return None

        source_profile = self.compound_profiles[source]['odor_profile']
        target_profile = self.compound_profiles[target]['odor_profile']
        similarity = np.dot(source_profile, target_profile) / (
            np.linalg.norm(source_profile) * np.linalg.norm(target_profile) + 1e-10)

        return {
            'source': source, 'target': target,
            'source_odors': self.compound_profiles[source]['dominant_odors'],
            'target_odors': self.compound_profiles[target]['dominant_odors'],
            'pathway_length': len(pathway_steps),
            'ec_sequence': ec_sequence,
            'odor_similarity': similarity,
            'steps': pathway_steps,
            'has_unknown_ec': 'unknown' in ec_sequence,
            'ec_coverage': sum(1 for s in pathway_steps if s['has_ec_data']) / len(pathway_steps)
        }

    def list_pathway_caches(self):
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
            print(f"Found pathway cache: {info['pathways']:,} pathways, "
                  f"max_length={info['max_length']}, created {info['created']}")
            return [info]
        except Exception as e:
            print(f"  ❌ Error reading cache: {e}")
            return []

    def clear_pathway_cache(self):
        cache_file = self.cache_dir / "pathways_cache.pkl"
        if cache_file.exists():
            try:
                cache_file.unlink()
                print("✅ Pathway cache cleared")
                return True
            except Exception as e:
                print(f"❌ Failed to clear cache: {e}")
                return False
        else:
            print("No pathway cache file to clear.")
            return True
