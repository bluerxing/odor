# Odor Transformation Rule Explorer

An interactive system for exploring enzymatic odor transformation rules mined from KEGG metabolic pathways.

## Live Demo

**[Open the Explorer →](https://bluerxing.github.io/odor/)**

## Features

- **Six Discoveries**: Pre-computed showcase of 6 structural findings about odor change (hard barriers, enzyme incompatibility, directional gates, sequence dominance, enzyme sufficiency, order sensitivity)
- **Interactive Network**: Full metabolic odor network visualization (vis.js)
- **Statistical Dashboard**: EC class distribution, source/target asymmetry, pathway lengths, rule types
- **Rule Query Engine**: Client-side query engine supporting 8 query types across 23,042 rules and 178,902 transformation events
- **Multi-Step Reasoning**: Counterfactual analysis, multi-step reachability, order sensitivity

## Data

- **92 odor categories** (from 138 controlled descriptors; 46 have no KEGG metabolic pathway)
- **25,554 metabolic pathways** from KEGG
- **178,902 odor transformation events**
- **23,042 logical rules** across 10 types (exclusion, necessity, sufficiency, incompatibility, etc.)

## Citation

If you use this system, please cite:

> [Paper citation TBD]

## License

MIT
