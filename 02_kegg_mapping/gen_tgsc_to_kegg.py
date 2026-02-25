#!/usr/bin/env python3
"""
Complete TGSC to KEGG mapper with kegg.txt parsing and resume
Input: curated_goodcents_with_tgsc.csv + kegg.txt (optional)
Output: tgsc_to_kegg.csv
"""
import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.thegoodscentscompany.com/data/"


def extract_kegg_ids(html: str):
    """Return all unique KEGG C-numbers found in a page."""
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    return list(dict.fromkeys(re.findall(r"\bC\d{5}\b", text)))


def fetch_page(rwid: str, session, delay=1.0):
    """Download one Good Scents page, respecting polite delay."""
    if not rwid.startswith('rw'):
        rwid = f"rw{rwid}"
    url = f"{BASE_URL}{rwid}.html"
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    time.sleep(delay)
    return resp.text


def parse_kegg_txt(txt_file: Path):
    """Parse kegg.txt file to get already processed results."""
    if not txt_file.exists():
        print(f"No {txt_file} found - starting fresh")
        return {}

    print(f"Parsing existing results from {txt_file}...")
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse entries like: "Processing X/Y: 1043411" followed by "  → KEGG IDs: C05771"
    pattern = r'Processing \d+/\d+: (\d+)\s*\n\s*→ KEGG IDs: ([^\n]*)'
    matches = re.findall(pattern, content)

    results = {}
    for tgsc_id, kegg_result in matches:
        kegg_result = kegg_result.strip()
        if kegg_result == "—":
            kegg_result = ""
        results[tgsc_id] = kegg_result

    print(f"Found {len(results)} already processed entries")
    return results


def resolve_input_csv(workdir: Path, explicit_path: str = None) -> Path:
    """Resolve input CSV path with backwards-compatible fallbacks."""
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = (workdir / path).resolve()
        return path

    candidates = [
        workdir / "curated_goodcents_with_tgsc.csv",
        workdir / "../01_data_preparation/curated_goodcents_with_tgsc.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TGSC to KEGG mapper")
    parser.add_argument(
        "--input",
        default=None,
        help="Input CSV path (default: auto-detect from 02_kegg_mapping or ../01_data_preparation)",
    )
    parser.add_argument("--output", default="tgsc_to_kegg.csv", help="Output CSV filename/path")
    parser.add_argument("--resume-log", default="kegg.txt", help="Resume log file (kegg.txt)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--save-interval", type=int, default=50, help="Autosave interval")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N rows (smoke test)")
    return parser


def main():
    args = build_arg_parser().parse_args()
    workdir = Path(__file__).resolve().parent

    csv_in = resolve_input_csv(workdir, args.input)
    csv_out = Path(args.output)
    txt_file = Path(args.resume_log)
    if not csv_out.is_absolute():
        csv_out = (workdir / csv_out).resolve()
    if not txt_file.is_absolute():
        txt_file = (workdir / txt_file).resolve()

    if not csv_in.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_in}\n"
            "Try --input ../01_data_preparation/curated_goodcents_with_tgsc.csv"
        )

    # Load input data
    df = pd.read_csv(csv_in, dtype=str)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
        print(f"⚠ Running in limited mode: first {len(df)} rows")
    tgsc_column = "TGSC ID"

    if tgsc_column not in df.columns:
        raise ValueError(f"CSV must contain '{tgsc_column}' column. Found: {list(df.columns)}")

    # Parse existing results from kegg.txt
    existing_results = parse_kegg_txt(txt_file)

    # Initialize KEGG_IDs column
    df["KEGG_IDs"] = ""

    # Apply existing results
    for idx, row in df.iterrows():
        tgsc_id = str(row[tgsc_column]).strip()
        if tgsc_id in existing_results:
            df.at[idx, "KEGG_IDs"] = existing_results[tgsc_id]

    # Count what we have
    processed_count = len(df[df["KEGG_IDs"].notna() & (df["KEGG_IDs"] != "")])
    total = len(df)

    print(f"Total entries: {total}")
    print(f"Already processed: {len(existing_results)}")
    print(f"Remaining to process: {total - len(existing_results)}")

    # Setup session
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (GoodScents-KEGG-Mapper/1.0)"

    # Process remaining entries
    processed_this_run = 0
    save_interval = max(1, args.save_interval)

    for idx, row in df.iterrows():
        tgsc_id = str(row[tgsc_column]).strip()

        # Skip if already processed
        if tgsc_id in existing_results:
            continue

        try:
            print(f"Processing {idx + 1}/{total}: {tgsc_id}")

            html = fetch_page(tgsc_id, session, delay=args.delay)
            kegg_ids = extract_kegg_ids(html)
            kegg_result = ";".join(kegg_ids)
            df.at[idx, "KEGG_IDs"] = kegg_result

            print(f"  → KEGG IDs: {kegg_result or '—'}")

            processed_this_run += 1

            # Save progress every 50 entries
            if processed_this_run % save_interval == 0:
                df.to_csv(csv_out, index=False)
                print(f"  [SAVED] Progress saved after {processed_this_run} new entries")

        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Saving progress...")
            df.to_csv(csv_out, index=False)
            print(f"✓ Saved {csv_out} with {processed_this_run} new entries processed")
            return

        except Exception as err:
            print(f"  [ERROR] {tgsc_id}: {err}")
            df.at[idx, "KEGG_IDs"] = "ERROR"
            processed_this_run += 1

    # Final save
    df.to_csv(csv_out, index=False)
    print(f"\n✓ Complete! Results saved to {csv_out}")

    # Summary
    with_kegg = len(df[(df["KEGG_IDs"] != "") & (df["KEGG_IDs"] != "ERROR")])
    empty = len(df[df["KEGG_IDs"] == ""])
    errors = len(df[df["KEGG_IDs"] == "ERROR"])

    print(f"✓ Found KEGG IDs: {with_kegg}")
    print(f"✓ No KEGG found: {empty}")
    print(f"✓ Errors: {errors}")
    print(f"✓ New entries processed this run: {processed_this_run}")


if __name__ == "__main__":
    main()
