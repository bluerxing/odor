# Odor Transformation Rules: KEGG Metabolic Network-based Discovery and Validation

Mining odor transformation rules from KEGG metabolic networks and validating them across embedding spaces and cross-species datasets.

## Pipeline Overview

```
01_data_preparation     GoodScents / Leffingwell dataset curation
        |
02_kegg_mapping         TGSC ID -> KEGG compound ID mapping
        |
03_rule_extraction      Odor network construction + pathway discovery + FOL rule extraction
        |
        +---------------------------+
        |                           |
04_validation_openpom       05_validation_cross_dataset
OpenPOM embedding space     Qian et al. 14 cross-species
geometric validation        datasets + 4 rule family validation
        |                           |
        +---------------------------+
                    |
          06_visualization
          Unified paper figures
                    |
          07_interactive_demo
          Browser-based query & reasoning demos
                    |
          08_paper
          LaTeX manuscript + figures (code -> figures -> paper)
```

## Directory Structure

```
final_version/
  01_data_preparation/
    goodcents_dataset_curation_keep_tgscid_FIXED.ipynb   # GoodScents data cleaning
    leffingwell_dataset_curation_with_cid_FIXED.ipynb    # Leffingwell data cleaning
    curated_goodcents_with_tgsc.csv                      # Cleaned GoodScents (4392 compounds)
    curated_leffingwell_with_cid.csv                     # Cleaned Leffingwell (3499 compounds)

  02_kegg_mapping/
    gen_tgsc_to_kegg.py        # Scrape GoodScents webpage -> KEGG C-numbers
    tgsc_to_kegg.csv           # Result: 644/4392 compounds with KEGG mapping

  03_rule_extraction/
    v5_all_with_vis.py         # Core pipeline (3270 lines):
                               #   load_tgsc_data -> download_kegg_data -> build_odor_network
                               #   -> find_odor_pathways -> expand_to_odor_events
                               #   -> ComplexRuleExtractor -> visualization

  04_validation_openpom/
    ec_rule_validation_v3.py   # EC operator direction consistency in OpenPOM embedding space
                               #   9 grouping strategies, permutation tests (n=1000)

  05_validation_cross_dataset/
    cross_dataset_ec_validation.py             # EC operator across Qian 14 cross-species datasets
    cross_dataset_ec_validation_multi_bake.py  # 4 rule families: compositional, counterfactual,
                                               #   gated, negative repulsion

  06_visualization/
    unified_visualization.py   # Paper-ready figures (dashboard, distributions, effect sizes)

  07_interactive_demo/
    odor_query_system.html         # Browser-based odor transformation query interface
    proof_reasoner_graph.html      # Real-time KEGG + Prolog graph reasoning demo

  08_paper/
    odor-rules-paper.tex           # Elsevier CAS single-column manuscript
    odor-refs.bib                  # BibTeX references
    cas-sc.cls                     # Elsevier document class
    cas-common.sty                 # Elsevier shared styles
    cas-model2-names.bst           # BibTeX style
    figures/                       # All paper figures (from visualization_charts + overview)
      overview.pdf                 #   Pipeline overview diagram
      11_empirical_dashboard.png   #   Composite dashboard (A-F panels from V5 outputs)
      01_data_overview.png         #   Data statistics
      02_odor_distribution.png     #   Odor category distribution
      03_pathway_length.png        #   Pathway length analysis
      04_transformation_heatmap.png#   Odor transformation heatmap
      05_top_ec_sequences.png      #   Top EC sequences
      07a_ec_by_source.png         #   EC grouped by source odor
      07b_ec_by_target.png         #   EC grouped by target odor
      08a_ec_function_pie_normalized.png  # EC function distribution
      09_case_studies.png          #   Case study pathways
      10_rule_boundaries.png       #   Rule boundary visualization

  kegg_cache/   -> symlink to ../kegg_cache (pathways_cache.pkl, rclass files)
  experiments/  -> symlink to ../experiments (checkpoint1.pt, OpenPOM weights)
  publications/ -> Qian et al. 2023 data (embeddings.csv, 14 datasets)
```

## Environment

- Python 3.9+
- Key dependencies:
  ```
  numpy, pandas, scipy, scikit-learn
  networkx
  torch, dgllife, dgl
  rdkit
  matplotlib, seaborn
  requests, beautifulsoup4
  ```

## Running the Pipeline

All commands should be run from within the corresponding subdirectory (paths are relative).

### One-command Orchestration (Recommended)
```bash
# Fast end-to-end verification
/opt/miniconda3/bin/python run_pipeline.py --mode smoke

# Full run (publication-level OpenPOM permutations)
/opt/miniconda3/bin/python run_pipeline.py --mode full
```

Useful switches:
```bash
# Re-scrape TGSC -> KEGG (slow, network required)
/opt/miniconda3/bin/python run_pipeline.py --mode smoke --run-kegg-mapping

# Skip expensive stages
/opt/miniconda3/bin/python run_pipeline.py --mode smoke --skip-openpom --skip-paper
```

### Stage 1: Data Preparation
```bash
# Open and run the Jupyter notebooks in 01_data_preparation/
# Output: curated_goodcents_with_tgsc.csv, curated_leffingwell_with_cid.csv
```

### Stage 2: KEGG Mapping
```bash
cd 02_kegg_mapping/
python gen_tgsc_to_kegg.py
# Output: tgsc_to_kegg.csv (already included, re-run only if needed)
# Note: requires internet access to scrape GoodScents
# Input auto-detected from:
#   02_kegg_mapping/curated_goodcents_with_tgsc.csv
#   or ../01_data_preparation/curated_goodcents_with_tgsc.csv
```

### Stage 3: Rule Extraction
```bash
cd 03_rule_extraction/
python v5_all_with_vis.py
# Output: pathways, odor events, FOL rules, network visualizations
# Also generates kegg_cache/pathways_cache.pkl (used by later stages)
```

### Stage 4: OpenPOM Validation
```bash
cd 04_validation_openpom/
python ec_rule_validation_v3.py
# Optional args: --ec-level 3 --n-permutations 1000 --min-group-size 5
# Output: ec_validation_results/ (JSON reports + PNG figures)
```

### Stage 5: Cross-dataset Validation
```bash
cd 05_validation_cross_dataset/

# Basic: EC operator cross-species validation
python cross_dataset_ec_validation.py

# Advanced: 4 rule family validation
python cross_dataset_ec_validation_multi_bake.py
# Output: cross_validation_report_multi.json, cross_validation_report_multi_raw.pkl
```

### Stage 6: Visualization
```bash
cd 06_visualization/
python unified_visualization.py
# Reads results from 05_validation_cross_dataset/
# Output: visualization_unified/ (paper-ready figures)
```

### Stage 7: Interactive Demos
```bash
# Open in browser directly — self-contained HTML, no server needed
open 07_interactive_demo/odor_query_system.html
open 07_interactive_demo/proof_reasoner_graph.html
```

### Stage 8: Paper Compilation
```bash
cd 08_paper/
# figures/ already contains all images referenced by the .tex
pdflatex odor-rules-paper.tex
bibtex odor-rules-paper
pdflatex odor-rules-paper.tex
pdflatex odor-rules-paper.tex
```

## External Data

- **publications/**: Qian et al. 2023 cross-species olfaction data
  - Source: `git clone https://github.com/osmoai/publications.git`
  - Contains embeddings.csv (OpenPOM 256-dim) and 14 species-specific datasets

- **experiments/checkpoint1.pt**: Pre-trained OpenPOM model weights

- **kegg_cache/**: Cached KEGG pathway data (generated by Stage 3)

## Code-Figures-Paper Workflow

The directory is designed for synchronized updates:

```
Code generates data    -->   Scripts produce figures   -->   Paper references figures
03_rule_extraction/         08_paper/figures/               08_paper/odor-rules-paper.tex
04_validation_openpom/        *.png, *.pdf                    \includegraphics{...}
05_validation_cross_dataset/
06_visualization/
```

**Typical update cycle:**
1. Modify analysis code (e.g., `03_rule_extraction/v5_all_with_vis.py`)
2. Re-run — outputs regenerate `visualization_charts/` images
3. Copy updated figures: `cp ../visualization_charts/*.{png,pdf} 08_paper/figures/`
4. Recompile paper: `cd 08_paper && pdflatex odor-rules-paper.tex`

**Figure-to-paper mapping** (all in `08_paper/figures/`):

| Figure in paper | File | Generated by |
|----------------|------|-------------|
| Fig 1 (overview) | `overview.pdf` | Manual / Inkscape |
| Fig 2 (rule boundaries) | `10_rule_boundaries.png` | Existing paper asset |
| Fig 3 (empirical dashboard) | `11_empirical_dashboard.png` | `v5_all_with_vis.py` |
| Fig 4 (EC function pie) | `08a_ec_function_pie_normalized.png` | `v5_all_with_vis.py` |
| Fig 5 (case studies) | `09_case_studies.png` | `v5_all_with_vis.py` |

## Key Design Decisions

1. **tgsc_to_kegg.csv over merged_dataset_with_kegg.csv**: The merged dataset suffers from stereo-isomer merging that pollutes odor descriptors (76% of merged rows have Jaccard < 0.5). Leffingwell contributes only 27 usable compounds via KEGG.

2. **ec_rule_validation_v3 over v3_bake**: v3 uses stricter parameters (min_group_size=5, n_permutations=1000, seed=42).

3. **unified_visualization.py**: Consolidates cross_dataset_visualization.py and visualization_multi_rule.py into one script.
