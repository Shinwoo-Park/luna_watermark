#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def find_existing_summary(cell_dir: Path, lang: str, dataset: str,
                          temperature: float) -> Path | None:
    pattern = f"summary_{lang}_{dataset}_*_T{temperature}.json"
    matches = list(cell_dir.glob(pattern))
    return matches[0] if matches else None


def run_cell(algo: str, lang: str, k: "int | None", args) -> tuple[bool, float]:
    suffix = f"_k{k}" if k is not None and algo in {"LUNA"} else ""
    cell_dir = Path(args.results_root) / f"{algo}_{lang}{suffix}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    existing = find_existing_summary(cell_dir, lang, args.dataset, args.temperature)
    if existing is not None and not args.force:
        print(f"  [SKIP] {algo} × {lang}{suffix} — summary exists: {existing.name}")
        return True, 0.0

    cmd = [
        sys.executable, "-u",
        "-m", "experiments.runners.run_experiment",
        "--algorithm", algo,
        "--lang", lang,
        "--dataset", args.dataset,
        "--temperature", str(args.temperature),
        "--num-samples", str(args.num_samples),
        "--max-detection-tokens", str(args.max_detection_tokens),
        "--data-dir", args.data_dir,
        "--output-dir", str(cell_dir),
    ]
    if k is not None:
        cmd += ["--k-primary", str(k)]

    label = f"{algo} × {lang}" + (f" k={k}" if k is not None else "")
    print(f"  ┌─ {label}  (dataset={args.dataset}, T={args.temperature}, N={args.num_samples})")
    print(f"  │  start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()

    t0 = time.time()
    rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    status = "✓" if rc == 0 else f"✗ (rc={rc})"
    mins = elapsed / 60
    print(f"  └─ {label}  {status}  ({mins:.1f}m)")
    print()
    sys.stdout.flush()
    return rc == 0, elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", nargs="+", required=True,
                   help="Cells to run, each as 'ALGO LANG'.")
    p.add_argument("--dataset", default="wiki", choices=("wiki", "news"),
                   help="Dataset family. Default: wiki.")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature. Default: 0.7.")
    p.add_argument("--num-samples", type=int, default=500,
                   help="Records per cell. Default: 500.")
    p.add_argument("--max-detection-tokens", type=int, default=256,
                   help="Detection-time token cap (matches MAX_NEW_TOKENS). Default: 256.")
    p.add_argument("--data-dir", default="data",
                   help="Directory containing {dataset}_{lang}.jsonl files.")
    p.add_argument("--results-root", required=True,
                   help="Output root, e.g. results/main_wiki_T0.7. Per-cell dirs "
                        "{ALGO}_{LANG}/ are created inside.")
    p.add_argument("--force", action="store_true",
                   help="Re-run cells even if their summary already exists.")
    args = p.parse_args()

    cells: list[tuple[str, str, "int | None"]] = []
    for s in args.pairs:
        parts = s.strip().split()
        if len(parts) == 2:
            cells.append((parts[0], parts[1], None))
        elif len(parts) == 3:
            try:
                k = int(parts[2])
            except ValueError:
                print(f"  bad k in --pairs entry: {s!r}", file=sys.stderr)
                sys.exit(2)
            cells.append((parts[0], parts[1], k))
        else:
            print(f"  bad --pairs entry: {s!r}  (expect 'ALGO LANG' or 'ALGO LANG K')",
                  file=sys.stderr)
            sys.exit(2)

    Path(args.results_root).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  run_main_chunk.py")
    print("=" * 70)
    print(f"  cells:        {len(cells)}")
    print(f"  dataset:      {args.dataset}")
    print(f"  temperature:  {args.temperature}")
    print(f"  num_samples:  {args.num_samples}")
    print(f"  results_root: {args.results_root}")
    print("=" * 70)
    print()

    n_pass = n_fail = n_skip = 0
    t_start = time.time()
    for i, (algo, lang, k) in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] elapsed so far: {(time.time()-t_start)/60:.1f}m")
        ok, elapsed = run_cell(algo, lang, k, args)
        if elapsed == 0.0 and ok:
            n_skip += 1
        elif ok:
            n_pass += 1
        else:
            n_fail += 1

    total = time.time() - t_start
    print("=" * 70)
    print(f"  chunk complete: {n_pass} ran, {n_skip} skipped, {n_fail} failed")
    print(f"  total elapsed:  {total/3600:.1f}h ({total/60:.1f}m)")
    print("=" * 70)

    if n_pass == 0 and n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
