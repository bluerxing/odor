#!/usr/bin/env python3
"""
End-to-end orchestrator for the clean odor-rules pipeline.

Default behavior is a smoke run:
- Reuses existing caches/results when possible
- Uses lightweight OpenPOM validation settings
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent
PAPER_DIR = ROOT / "08_paper"


def pick_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    preferred = ["/opt/miniconda3/bin/python", sys.executable]
    for candidate in preferred:
        if Path(candidate).exists():
            return candidate
    return sys.executable


def run_cmd(cmd: Sequence[str], cwd: Path, env: dict) -> None:
    cmd_str = " ".join(cmd)
    print(f"\n[RUN] {cwd.relative_to(ROOT)} :: {cmd_str}", flush=True)
    subprocess.run(list(cmd), cwd=cwd, env=env, check=True)


def sync_rule_figures() -> None:
    src_dir = ROOT / "03_rule_extraction" / "visualization_charts"
    dst_dir = ROOT / "08_paper" / "figures"
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied_png = 0
    copied_pdf = 0

    for src in sorted(src_dir.glob("*.png")):
        shutil.copy2(src, dst_dir / src.name)
        copied_png += 1

    for src in sorted(src_dir.glob("*.pdf")):
        shutil.copy2(src, dst_dir / src.name)
        copied_pdf += 1

    print(
        f"[SYNC] Copied {copied_png} PNG and {copied_pdf} PDF files to 08_paper/figures",
        flush=True,
    )


def compile_paper(env: dict) -> None:
    run_cmd(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "odor-rules-paper.tex"],
        PAPER_DIR,
        env,
    )
    run_cmd(["bibtex", "odor-rules-paper"], PAPER_DIR, env)
    run_cmd(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "odor-rules-paper.tex"],
        PAPER_DIR,
        env,
    )
    run_cmd(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "odor-rules-paper.tex"],
        PAPER_DIR,
        env,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the odor-rules pipeline")
    parser.add_argument("--python", default=None, help="Python interpreter path")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="smoke: fast verification, full: publication-level OpenPOM settings",
    )
    parser.add_argument(
        "--run-kegg-mapping",
        action="store_true",
        help="Re-run TGSC->KEGG scraping stage (requires internet and can be slow)",
    )
    parser.add_argument("--skip-openpom", action="store_true", help="Skip stage 04 validation")
    parser.add_argument("--skip-cross-basic", action="store_true", help="Skip basic cross-dataset validation")
    parser.add_argument("--skip-cross-multi", action="store_true", help="Skip multi-rule validation")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip unified visualization")
    parser.add_argument("--skip-paper", action="store_true", help="Skip LaTeX compilation")
    parser.add_argument("--no-sync-figures", action="store_true", help="Do not copy stage-03 figures to 08_paper")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python_bin = pick_python(args.python)

    env = os.environ.copy()
    mpl_dir = ROOT / ".mplconfig"
    mpl_dir.mkdir(exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env.setdefault("MPLBACKEND", "Agg")

    print("=" * 80, flush=True)
    print("Odor Rules Pipeline Orchestrator", flush=True)
    print(f"Root: {ROOT}", flush=True)
    print(f"Python: {python_bin}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print("=" * 80, flush=True)

    try:
        if args.run_kegg_mapping:
            run_cmd([python_bin, "gen_tgsc_to_kegg.py"], ROOT / "02_kegg_mapping", env)

        run_cmd([python_bin, "v5_all_with_vis.py"], ROOT / "03_rule_extraction", env)

        if not args.skip_openpom:
            if args.mode == "smoke":
                openpom_cmd = [
                    python_bin,
                    "ec_rule_validation_v3.py",
                    "--n-permutations",
                    "20",
                    "--max-samples",
                    "2000",
                    "--output-dir",
                    "./ec_validation_results_smoke",
                ]
            else:
                openpom_cmd = [
                    python_bin,
                    "ec_rule_validation_v3.py",
                    "--n-permutations",
                    "1000",
                    "--max-samples",
                    "20000",
                    "--output-dir",
                    "./ec_validation_results",
                ]
            run_cmd(openpom_cmd, ROOT / "04_validation_openpom", env)

        if not args.skip_cross_basic:
            run_cmd([python_bin, "cross_dataset_ec_validation.py"], ROOT / "05_validation_cross_dataset", env)

        if not args.skip_cross_multi:
            run_cmd(
                [python_bin, "cross_dataset_ec_validation_multi_bake.py"],
                ROOT / "05_validation_cross_dataset",
                env,
            )

        if not args.skip_visualization:
            run_cmd([python_bin, "unified_visualization.py"], ROOT / "06_visualization", env)

        if not args.no_sync_figures:
            sync_rule_figures()

        if not args.skip_paper:
            compile_paper(env)

    except subprocess.CalledProcessError as exc:
        print("\n[FAILED]", flush=True)
        print(f"Command: {' '.join(exc.cmd)}", flush=True)
        print(f"Exit code: {exc.returncode}", flush=True)
        return exc.returncode

    print("\n[SUCCESS] Pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
