#!/usr/bin/env python3
"""
Run a unified MedAgentSim defense benchmark suite across:
- Open Prompt Injection style patient/observation surfaces
- ASB-style attack families

This script reuses the existing benchmark runners, then exports compact
comparison artifacts so different defenses can be compared side by side.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


VALID_OPENPI_SURFACES = ("patient", "observation")
VALID_ASB_FAMILIES = ("dpi", "opi", "memory", "mixed", "pot_backdoor")
VALID_ATTACKS = ("naive", "ignore", "escape", "fake_comp", "combine")
VALID_TIMINGS = ("early", "late")
DEFAULT_DEFENSES = (
    "none,known_answer,llm_based,layered_guard,"
    "response_based,structured_guard,prompt_guard,ppl-10-4.5"
)


def _repo_paths() -> tuple[Path, Path, Path]:
    script = Path(__file__).resolve()
    medsim_root = script.parents[1]
    security_runner = medsim_root / "scripts" / "run_medqa_security_benchmark.py"
    asb_runner = medsim_root / "scripts" / "run_medqa_asb_benchmark.py"
    return medsim_root, security_runner, asb_runner


def parse_csv(raw: str, valid: tuple[str, ...] | None = None) -> list[str]:
    items = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if valid is not None:
        bad = [item for item in items if item not in valid]
        if bad:
            raise ValueError(f"Invalid values: {bad}; allowed: {valid}")
    return items


def run_cmd(argv: list[str], cwd: Path, dry_run: bool = False) -> int:
    print("+", " ".join(argv), flush=True)
    if dry_run:
        return 0
    result = subprocess.run(argv, cwd=str(cwd), env=os.environ.copy())
    return int(result.returncode)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def weighted_average(rows: list[dict[str, Any]], key: str, weight_key: str = "N") -> float | None:
    total_weight = 0.0
    total_value = 0.0
    for row in rows:
        value = safe_float(row.get(key))
        weight = safe_float(row.get(weight_key))
        if value is None or weight is None or weight <= 0:
            continue
        total_value += value * weight
        total_weight += weight
    if total_weight <= 0:
        return None
    return total_value / total_weight


def ratio_from_weighted_counts(
    rows: list[dict[str, Any]],
    rate_key: str,
    denom_key: str,
) -> float | None:
    denom = 0.0
    numer = 0.0
    for row in rows:
        rate = safe_float(row.get(rate_key))
        count = safe_float(row.get(denom_key))
        if rate is None or count is None or count <= 0:
            continue
        numer += rate * count
        denom += count
    if denom <= 0:
        return None
    return numer / denom


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_md_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(format_md_value(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_openpi_summary(summary_path: Path, defense: str, surface: str) -> dict[str, Any]:
    payload = load_json(summary_path)
    rows = list(payload.get("summaries") or [])
    total_cases = int(sum(int(row.get("N", 0) or 0) for row in rows))
    benign_turns = int(sum(int(row.get("N_defense_turns_benign", 0) or 0) for row in rows))
    malicious_turns = int(sum(int(row.get("N_defense_turns_malicious", 0) or 0) for row in rows))

    return {
        "benchmark": "openpi",
        "surface": surface,
        "defense": defense,
        "groups": len(rows),
        "total_cases": total_cases,
        "avg_PNA_T": round_or_none(weighted_average(rows, "PNA_T")),
        "avg_PNA_I": round_or_none(weighted_average(rows, "PNA_I")),
        "avg_ASV": round_or_none(weighted_average(rows, "ASV")),
        "avg_MR": round_or_none(weighted_average(rows, "MR")),
        "avg_baseline_accuracy": round_or_none(weighted_average(rows, "baseline_accuracy")),
        "avg_attack_accuracy": round_or_none(weighted_average(rows, "attack_accuracy")),
        "avg_accuracy_drop": round_or_none(weighted_average(rows, "accuracy_drop")),
        "avg_diagnosis_change_rate": round_or_none(weighted_average(rows, "diagnosis_change_rate")),
        "avg_FPR": round_or_none(ratio_from_weighted_counts(rows, "FPR", "N_defense_turns_benign")),
        "avg_FNR": round_or_none(ratio_from_weighted_counts(rows, "FNR", "N_defense_turns_malicious")),
        "defense_turns_benign": benign_turns,
        "defense_turns_malicious": malicious_turns,
        "summary_json": str(summary_path),
    }


def summarize_asb_summary(summary_path: Path, defense: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(summary_path)
    rows = list(payload.get("family_summaries") or [])
    average_row = next((row for row in rows if row.get("Attack Family") == "Average"), None)
    family_rows = [row for row in rows if row.get("Attack Family") != "Average"]
    if average_row is None:
        raise ValueError(f"Missing Average row in ASB summary: {summary_path}")

    overall = {
        "benchmark": "asb",
        "defense": defense,
        "total_runs": int(average_row.get("N", 0) or 0),
        "avg_ASR": round_or_none(safe_float(average_row.get("ASR"))),
        "avg_original_task_success_rate": round_or_none(
            safe_float(average_row.get("Original task success rate"))
        ),
        "avg_RR": round_or_none(safe_float(average_row.get("RR"))),
        "successful_attack_num": int(average_row.get("Successful attack num", 0) or 0),
        "original_task_success_num": int(average_row.get("Original task success num", 0) or 0),
        "refuse_num": int(average_row.get("Refuse number", 0) or 0),
        "summary_json": str(summary_path),
    }

    family_details: list[dict[str, Any]] = []
    for row in family_rows:
        family_details.append(
            {
                "benchmark": "asb",
                "defense": defense,
                "family": row.get("Attack Family", ""),
                "N": int(row.get("N", 0) or 0),
                "ASR": round_or_none(safe_float(row.get("ASR"))),
                "Original task success rate": round_or_none(
                    safe_float(row.get("Original task success rate"))
                ),
                "RR": round_or_none(safe_float(row.get("RR"))),
                "Successful attack num": int(row.get("Successful attack num", 0) or 0),
                "Original task success num": int(row.get("Original task success num", 0) or 0),
                "Refuse number": int(row.get("Refuse number", 0) or 0),
                "summary_json": str(summary_path),
            }
        )
    return overall, family_details


def build_markdown_report(
    *,
    openpi_rows_by_surface: dict[str, list[dict[str, Any]]],
    asb_overall_rows: list[dict[str, Any]],
    asb_family_rows: list[dict[str, Any]],
    suite_root: Path,
    openpi_reports_root: Path,
    asb_reports_root: Path,
) -> str:
    lines = [
        "# MedAgentSim Defense Benchmark Suite",
        "",
        "This report combines Open Prompt Injection style results and ASB-style family results.",
        "",
        "Interpretation guide:",
        "- OpenPI: lower `avg_ASV`, `avg_accuracy_drop`, `avg_FPR`, and `avg_FNR` is better.",
        "- ASB: lower `avg_ASR` is better, while higher `avg_original_task_success_rate` preserves utility.",
        "- `avg_RR` is useful for safety comparison, but a higher refusal rate may also mean more over-defense.",
        "",
        f"- Suite root: `{suite_root}`",
        f"- OpenPI root: `{openpi_reports_root}`",
        f"- ASB root: `{asb_reports_root}`",
        "",
    ]

    for surface, rows in openpi_rows_by_surface.items():
        lines.extend(
            [
                f"## OpenPI {surface.title()}",
                "",
                markdown_table(
                    rows,
                    [
                        "defense",
                        "avg_ASV",
                        "avg_accuracy_drop",
                        "avg_FPR",
                        "avg_FNR",
                        "avg_attack_accuracy",
                        "total_cases",
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## ASB Overall",
            "",
            markdown_table(
                asb_overall_rows,
                [
                    "defense",
                    "avg_ASR",
                    "avg_original_task_success_rate",
                    "avg_RR",
                    "total_runs",
                ],
            ),
            "",
        ]
    )

    if asb_family_rows:
        lines.extend(
            [
                "## ASB Families",
                "",
                markdown_table(
                    asb_family_rows,
                    [
                        "defense",
                        "family",
                        "ASR",
                        "Original task success rate",
                        "RR",
                        "N",
                    ],
                ),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    medsim_root, _, _ = _repo_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="smoke", choices=("smoke", "full"))
    parser.add_argument("--defenses", default=DEFAULT_DEFENSES)
    parser.add_argument("--openpi-surfaces", default="patient,observation")
    parser.add_argument("--asb-families", default="dpi,opi,memory,mixed,pot_backdoor")
    parser.add_argument("--attacks", default="naive,ignore,escape,fake_comp,combine")
    parser.add_argument("--timings", default="early,late")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--total-inferences", type=int, default=10)
    parser.add_argument(
        "--cases-file",
        default=str(medsim_root / "scripted_inputs_medqa" / "medqa_all_107_cases.json"),
    )
    parser.add_argument(
        "--script-input-dir",
        default=str(medsim_root / "scripted_inputs_medqa"),
    )
    parser.add_argument(
        "--suite-root",
        default=str(medsim_root / "output_eval_medqa_defense_suite"),
    )
    parser.add_argument("--doctor-llm", default=os.environ.get("DOCTOR_LLM", os.environ.get("REMOTE_LLM_MODEL", "Qwen3.5-27B-Q4_K_M.gguf")))
    parser.add_argument("--measurement-llm", default=os.environ.get("MEASUREMENT_LLM", ""))
    parser.add_argument("--moderator-llm", default=os.environ.get("MODERATOR_LLM", ""))
    parser.add_argument("--doctor-image-request", default=os.environ.get("DOCTOR_IMAGE_REQUEST", "False"))
    parser.add_argument("--global-target", default=os.environ.get("GLOBAL_TARGET", ""))
    parser.add_argument(
        "--include-injected-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward to the OpenPI benchmark so PNA-I and MR can be computed.",
    )
    parser.add_argument(
        "--run-openpi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run or summarize OpenPI patient/observation benchmarks.",
    )
    parser.add_argument(
        "--run-asb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run or summarize ASB-style family benchmarks.",
    )
    parser.add_argument(
        "--run-benchmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Execute the benchmark runners before exporting comparison artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    medsim_root, security_runner, asb_runner = _repo_paths()
    suite_root = Path(args.suite_root).resolve()
    suite_root.mkdir(parents=True, exist_ok=True)
    comparison_root = suite_root / "comparison"
    openpi_root = suite_root / "openpi"
    asb_root = suite_root / "asb"

    defenses = [item.strip() for item in args.defenses.split(",") if item.strip()]
    if not defenses:
        raise ValueError("At least one defense must be provided.")
    openpi_surfaces = parse_csv(args.openpi_surfaces, VALID_OPENPI_SURFACES)
    asb_families = parse_csv(args.asb_families, VALID_ASB_FAMILIES)
    attacks = parse_csv(args.attacks, VALID_ATTACKS)
    timings = parse_csv(args.timings, VALID_TIMINGS)

    if args.run_openpi:
        argv = [
            sys.executable,
            str(security_runner),
            "--preset",
            args.preset,
            "--surfaces",
            ",".join(openpi_surfaces),
            "--defenses",
            ",".join(defenses),
            "--attacks",
            ",".join(attacks),
            "--timings",
            ",".join(timings),
            "--batch-size",
            str(args.batch_size),
            "--total-inferences",
            str(args.total_inferences),
            "--cases-file",
            str(Path(args.cases_file).resolve()),
            "--script-input-dir",
            str(Path(args.script_input_dir).resolve()),
            "--reports-root",
            str(openpi_root),
            "--doctor-llm",
            args.doctor_llm,
            "--measurement-llm",
            args.measurement_llm or args.doctor_llm,
            "--moderator-llm",
            args.moderator_llm or args.doctor_llm,
            "--doctor-image-request",
            args.doctor_image_request,
        ]
        if args.global_target.strip():
            argv += ["--global-target", args.global_target.strip()]
        argv.append("--include-injected-only" if args.include_injected_only else "--no-include-injected-only")
        argv.append("--run-benchmark" if args.run_benchmarks else "--no-run-benchmark")
        if args.dry_run:
            argv.append("--dry-run")
        rc = run_cmd(argv, cwd=medsim_root, dry_run=False)
        if rc != 0:
            return rc

    if args.run_asb:
        argv = [
            sys.executable,
            str(asb_runner),
            "--preset",
            args.preset,
            "--families",
            ",".join(asb_families),
            "--defenses",
            ",".join(defenses),
            "--attacks",
            ",".join(attacks),
            "--timings",
            ",".join(timings),
            "--batch-size",
            str(args.batch_size),
            "--total-inferences",
            str(args.total_inferences),
            "--cases-file",
            str(Path(args.cases_file).resolve()),
            "--reports-root",
            str(asb_root),
            "--doctor-llm",
            args.doctor_llm,
            "--measurement-llm",
            args.measurement_llm or args.doctor_llm,
            "--moderator-llm",
            args.moderator_llm or args.doctor_llm,
            "--doctor-image-request",
            args.doctor_image_request,
        ]
        if args.global_target.strip():
            argv += ["--global-target", args.global_target.strip()]
        argv.append("--run-benchmark" if args.run_benchmarks else "--no-run-benchmark")
        if args.dry_run:
            argv.append("--dry-run")
        rc = run_cmd(argv, cwd=medsim_root, dry_run=False)
        if rc != 0:
            return rc

    if args.dry_run:
        print(
            json.dumps(
                {
                    "suite_root": str(suite_root),
                    "openpi_root": str(openpi_root),
                    "asb_root": str(asb_root),
                    "defenses": defenses,
                    "openpi_surfaces": openpi_surfaces,
                    "asb_families": asb_families,
                    "attacks": attacks,
                    "timings": timings,
                    "note": "Dry run only: comparison artifacts are not generated because benchmark outputs are not written.",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    openpi_rows_by_surface: dict[str, list[dict[str, Any]]] = {surface: [] for surface in openpi_surfaces}
    asb_overall_rows: list[dict[str, Any]] = []
    asb_family_rows: list[dict[str, Any]] = []

    if args.run_openpi:
        for surface in openpi_surfaces:
            for defense in defenses:
                summary_path = openpi_root / "summaries" / f"{surface}_{defense}.json"
                if not summary_path.is_file():
                    raise FileNotFoundError(f"Missing OpenPI summary: {summary_path}")
                openpi_rows_by_surface[surface].append(
                    summarize_openpi_summary(summary_path, defense=defense, surface=surface)
                )
            openpi_rows_by_surface[surface].sort(
                key=lambda row: (
                    safe_float(row.get("avg_ASV")) if safe_float(row.get("avg_ASV")) is not None else float("inf"),
                    safe_float(row.get("avg_accuracy_drop")) if safe_float(row.get("avg_accuracy_drop")) is not None else float("inf"),
                    safe_float(row.get("avg_FNR")) if safe_float(row.get("avg_FNR")) is not None else float("inf"),
                    str(row.get("defense", "")),
                )
            )

    if args.run_asb:
        for defense in defenses:
            summary_path = asb_root / "summaries" / f"asb_{defense}.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing ASB summary: {summary_path}")
            overall, families = summarize_asb_summary(summary_path, defense=defense)
            asb_overall_rows.append(overall)
            asb_family_rows.extend(families)
        asb_overall_rows.sort(
            key=lambda row: (
                safe_float(row.get("avg_ASR")) if safe_float(row.get("avg_ASR")) is not None else float("inf"),
                -(safe_float(row.get("avg_original_task_success_rate")) or 0.0),
                str(row.get("defense", "")),
            )
        )
        asb_family_rows.sort(key=lambda row: (str(row.get("family", "")), safe_float(row.get("ASR")) or float("inf"), str(row.get("defense", ""))))

    comparison_root.mkdir(parents=True, exist_ok=True)
    for surface, rows in openpi_rows_by_surface.items():
        if rows:
            write_csv(comparison_root / f"openpi_{surface}_comparison.csv", rows)
    if asb_overall_rows:
        write_csv(comparison_root / "asb_overall_comparison.csv", asb_overall_rows)
    if asb_family_rows:
        write_csv(comparison_root / "asb_family_comparison.csv", asb_family_rows)

    summary_payload = {
        "suite_root": str(suite_root),
        "openpi_reports_root": str(openpi_root),
        "asb_reports_root": str(asb_root),
        "defenses": defenses,
        "openpi_surfaces": openpi_surfaces,
        "asb_families": asb_families,
        "attacks": attacks,
        "timings": timings,
        "comparison_files": {
            "openpi": {
                surface: str(comparison_root / f"openpi_{surface}_comparison.csv")
                for surface, rows in openpi_rows_by_surface.items()
                if rows
            },
            "asb_overall": str(comparison_root / "asb_overall_comparison.csv") if asb_overall_rows else "",
            "asb_family": str(comparison_root / "asb_family_comparison.csv") if asb_family_rows else "",
            "report_md": str(comparison_root / "defense_suite_report.md"),
        },
        "openpi_ranked_rows": openpi_rows_by_surface,
        "asb_overall_ranked_rows": asb_overall_rows,
        "asb_family_rows": asb_family_rows,
        "notes": {
            "openpi_sort_order": "sorted by avg_ASV asc, then avg_accuracy_drop asc, then avg_FNR asc",
            "asb_sort_order": "sorted by avg_ASR asc, then avg_original_task_success_rate desc",
            "ranking_caution": "No single scalar score is imposed because safer defenses may trade off refusal rate and task utility differently.",
        },
    }
    summary_json = comparison_root / "defense_suite_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_md = build_markdown_report(
        openpi_rows_by_surface=openpi_rows_by_surface,
        asb_overall_rows=asb_overall_rows,
        asb_family_rows=asb_family_rows,
        suite_root=suite_root,
        openpi_reports_root=openpi_root,
        asb_reports_root=asb_root,
    )
    (comparison_root / "defense_suite_report.md").write_text(report_md, encoding="utf-8")

    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
