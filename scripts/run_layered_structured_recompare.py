#!/usr/bin/env python3
"""
Re-run only structured_guard vs layered_guard after defense changes,
then export a side-by-side comparison against the previous suite (if present).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _paths() -> tuple[Path, Path, Path, Path]:
    medsim_root = Path(__file__).resolve().parents[1]
    suite_runner = medsim_root / "scripts" / "run_medqa_defense_suite.py"
    old_suite = medsim_root / "output_eval_medqa_defense_suite"
    new_suite = medsim_root / "output_eval_medqa_layered_v2_compare"
    return medsim_root, suite_runner, old_suite, new_suite


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key not in row or row[key] is None:
            continue
        raw = row[key]
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                return raw
        elif isinstance(raw, (int, float)):
            value = float(raw)
        else:
            return str(raw)
        if 0.0 <= value <= 1.0:
            return f"{value * 100:.2f}%"
        return f"{value:.4f}"
    return "-"


def print_comparison(old_root: Path, new_root: Path) -> None:
    lines: list[str] = []
    lines.append("## So sánh structured_guard vs layered_guard (trước / sau sửa code)")
    lines.append("")
    lines.append("| Benchmark | Chỉ số | structured (cũ) | layered (cũ) | structured (mới) | layered (mới) |")
    lines.append("|-----------|--------|-----------------|--------------|------------------|---------------|")

    def row(bench_label: str, csv_name: str, metric_name: str, keys: tuple[str, ...]) -> None:
        old_path = old_root / "comparison" / csv_name
        new_path = new_root / "comparison" / csv_name
        old_vals = _read_csv_by_defense(old_path) if old_path.is_file() else {}
        new_vals = _read_csv_by_defense(new_path) if new_path.is_file() else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    bench_label,
                    metric_name,
                    _metric(old_vals.get("structured_guard", {}), *keys),
                    _metric(old_vals.get("layered_guard", {}), *keys),
                    _metric(new_vals.get("structured_guard", {}), *keys),
                    _metric(new_vals.get("layered_guard", {}), *keys),
                ]
            )
            + " |"
        )

    row("OpenPI patient", "openpi_patient_comparison.csv", "ASV", ("avg_ASV", "ASV"))
    row("OpenPI patient", "openpi_patient_comparison.csv", "FPR", ("avg_FPR", "FPR"))
    row("ASB", "asb_overall_comparison.csv", "ASR", ("avg_ASR", "ASR"))
    row("MPIB", "mpib_v1_overall_comparison.csv", "target_match", ("target_match_rate",))
    row("MPIB", "mpib_v1_overall_comparison.csv", "accuracy_drop", ("accuracy_drop",))

    ranking_old = old_root / "comparison" / "overall_defense_ranking.csv"
    ranking_new = new_root / "comparison" / "overall_defense_ranking.csv"
    if ranking_old.is_file() or ranking_new.is_file():
        lines.append("")
        lines.append("### mean_rank (balanced)")
        lines.append("")
        lines.append("| Defense | Cũ | Mới |")
        lines.append("|---------|-----|-----|")
        for defense in ("structured_guard", "layered_guard"):
            old_rank = _rank_from_csv(ranking_old, defense) if ranking_old.is_file() else "-"
            new_rank = _rank_from_csv(ranking_new, defense) if ranking_new.is_file() else "-"
            lines.append(f"| {defense} | {old_rank} | {new_rank} |")

    lines.append("")
    lines.append("### Ghi chú")
    lines.append("")
    lines.append("- **Cũ:** `output_eval_medqa_defense_suite` (code chưa tách ablation).")
    lines.append("- **Mới:** `output_eval_medqa_layered_v2_compare` (sau sửa layered_guard v2).")
    lines.append("- Commit report: copy toàn bộ thư mục `comparison/` vào repo sau khi chạy xong trên server.")

    report = new_root / "comparison" / "layered_structured_recompare.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {report}")


def _read_csv_by_defense(path: Path) -> dict[str, dict[str, str]]:
    import csv

    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            defense = (item.get("defense") or "").strip()
            if defense:
                rows[defense] = item
    return rows


def _rank_from_csv(path: Path, defense: str, profile: str = "balanced") -> str:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            if (item.get("defense") or "").strip() != defense:
                continue
            if profile and (item.get("ranking_profile") or "").strip() != profile:
                continue
            return item.get("mean_rank", "-")
    return "-"


def main() -> int:
    medsim_root, suite_runner, old_suite, new_suite = _paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="smoke", choices=("smoke", "full"))
    parser.add_argument(
        "--defenses",
        default="structured_guard,layered_guard",
        help="Only re-run changed defenses (default: structured_guard,layered_guard).",
    )
    parser.add_argument(
        "--suite-root",
        default=str(new_suite),
        help="Output directory for the new comparison run.",
    )
    parser.add_argument(
        "--old-suite-root",
        default=str(old_suite),
        help="Previous full suite root for before/after table.",
    )
    parser.add_argument("--run-mpib", action="store_true", help="Include MPIB-V1 (slower).")
    parser.add_argument(
        "--offline-replay",
        action="store_true",
        help="Skip LLM benchmarks; replay L2/L4 on existing suite dialogues only.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.offline_replay:
        replay_script = medsim_root / "scripts" / "replay_defense_layer_v2.py"
        replay_argv = [
            sys.executable,
            str(replay_script),
            "--old-suite-root",
            args.old_suite_root,
            "--out-dir",
            str(Path(args.suite_root) / "comparison"),
        ]
        print("+", " ".join(replay_argv), flush=True)
        if not args.dry_run:
            result = subprocess.run(replay_argv, cwd=str(medsim_root))
            if result.returncode != 0:
                return int(result.returncode)
            print_comparison(Path(args.old_suite_root), Path(args.suite_root))
        return 0

    argv = [
        sys.executable,
        str(suite_runner),
        "--preset",
        args.preset,
        "--defenses",
        args.defenses,
        "--suite-root",
        args.suite_root,
        "--run-benchmarks",
    ]
    if args.run_mpib:
        argv.append("--run-mpib")

    print("+", " ".join(argv), flush=True)
    if not args.dry_run:
        result = subprocess.run(argv, cwd=str(medsim_root))
        if result.returncode != 0:
            return int(result.returncode)

    if not args.dry_run:
        print_comparison(Path(args.old_suite_root), Path(args.suite_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
