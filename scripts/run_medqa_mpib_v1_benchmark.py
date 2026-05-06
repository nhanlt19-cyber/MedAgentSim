#!/usr/bin/env python3
"""
Run a multi-defense MPIB-V1 benchmark for MedAgentSim.

This wrapper keeps the attack source fixed to the generated MPIB-V1
manifest/scripts, then benchmarks one or more prompt-injection defenses and
exports comparison tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


VALID_RULES = ("R1", "R2", "R3", "R4", "R5", "R6")
VALID_TIMINGS = ("early", "late")
VALID_TIERS = ("strict", "realistic")
VALID_SCENARIO_FAMILIES = ("S1", "S2", "S3", "S4")
DEFAULT_DEFENSES = (
    "none,known_answer,llm_based,layered_guard,"
    "response_based,structured_guard,prompt_guard,ppl-10-4.5"
)


def _repo_paths() -> tuple[Path, Path, Path, Path]:
    script = Path(__file__).resolve()
    medsim_root = script.parents[1]
    generator = medsim_root / "scripts" / "generate_medqa_mpib_v1_dataset.py"
    runner = medsim_root / "scripts" / "run_medqa_mpib_v1_batch_resume.py"
    summarizer = medsim_root / "scripts" / "summarize_medqa_mpib_v1_results.py"
    return medsim_root, generator, runner, summarizer


def parse_csv(raw: str, valid: tuple[str, ...] | None = None) -> list[str]:
    items = [item.strip() for item in (raw or "").split(",") if item.strip()]
    if valid is not None:
        invalid = [item for item in items if item not in valid]
        if invalid:
            raise ValueError(f"Invalid values: {invalid}; allowed: {valid}")
    return items


def run_cmd(argv: list[str], cwd: Path, dry_run: bool = False) -> int:
    print("+", " ".join(argv), flush=True)
    if dry_run:
        return 0
    result = subprocess.run(argv, cwd=str(cwd), env=os.environ.copy())
    return int(result.returncode)


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def filter_manifest_rows(
    rows: list[dict[str, Any]],
    *,
    rules: list[str],
    timings: list[str],
    tiers: list[str],
    scenario_families: list[str],
    scenarios: set[int] | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        try:
            scenario_id = int(row["scenario_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if scenarios is not None and scenario_id not in scenarios:
            continue
        if rules and row.get("rule_family_id") not in rules:
            continue
        if timings and row.get("timing") not in timings:
            continue
        if tiers and row.get("tier") not in tiers:
            continue
        if scenario_families and row.get("scenario_family") not in scenario_families:
            continue
        filtered.append(row)
    return filtered


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def build_markdown_report(
    *,
    overall_rows: list[dict[str, Any]],
    by_rule_rows: list[dict[str, Any]],
    suite_root: Path,
    manifest_path: Path,
) -> str:
    lines = [
        "# MedAgentSim MPIB-V1 Defense Benchmark",
        "",
        "This report compares multiple defenses while keeping the attack data fixed to the MPIB-V1 manifest/scripts.",
        "",
        f"- Suite root: `{suite_root}`",
        f"- Manifest: `{manifest_path}`",
        "",
        "Interpretation guide:",
        "- Higher `attack_accuracy` / `robust_accuracy` is better.",
        "- Lower `target_match_rate` and `diagnosis_change_rate` is better.",
        "- Lower observed harm rates and `observed_cher` is better.",
        "",
        "## Overall By Defense",
        "",
        markdown_table(
            overall_rows,
            [
                "defense",
                "N",
                "attack_accuracy",
                "target_match_rate",
                "diagnosis_change_rate",
                "observed_cher",
                "observed_severity_mean",
            ],
        ),
        "",
    ]

    if by_rule_rows:
        lines.extend(
            [
                "## By Rule",
                "",
                markdown_table(
                    by_rule_rows,
                    [
                        "defense",
                        "rule_family_id",
                        "tier",
                        "timing",
                        "N",
                        "attack_accuracy",
                        "target_match_rate",
                        "diagnosis_change_rate",
                        "observed_cher",
                    ],
                ),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    medsim_root, _, _, _ = _repo_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="smoke", choices=("smoke", "full"))
    parser.add_argument("--defenses", default=DEFAULT_DEFENSES)
    parser.add_argument(
        "--manifest",
        default=str(medsim_root / "scripted_inputs_medqa_mpib" / "medqa_mpib_v1_manifest.jsonl"),
    )
    parser.add_argument(
        "--script-input-dir",
        default=str(medsim_root / "scripted_inputs_medqa_mpib"),
        help="Root directory containing generated MPIB-V1 attack scripts referenced by the manifest.",
    )
    parser.add_argument(
        "--baseline-script-dir",
        default=str(medsim_root / "scripted_inputs_medqa"),
    )
    parser.add_argument(
        "--source-cases",
        default=str(medsim_root / "scripted_inputs_medqa" / "medqa_all_107_cases.json"),
    )
    parser.add_argument(
        "--medqa-jsonl",
        default=str(medsim_root / "datasets" / "_medqa.jsonl"),
    )
    parser.add_argument(
        "--case-index",
        default=str(medsim_root / "scripted_inputs_medqa_mpib" / "medqa_mpib_case_index.json"),
    )
    parser.add_argument(
        "--reports-root",
        default=str(medsim_root / "output_eval_medqa_mpib_v1_benchmark"),
    )
    parser.add_argument("--rules", default="")
    parser.add_argument("--timings", default="")
    parser.add_argument("--tiers", default="")
    parser.add_argument("--scenario-families", default="")
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--total-inferences", type=int, default=10)
    parser.add_argument("--doctor-llm", default=os.environ.get("DOCTOR_LLM", os.environ.get("REMOTE_LLM_MODEL", "Qwen3.5-27B-Q4_K_M.gguf")))
    parser.add_argument("--measurement-llm", default=os.environ.get("MEASUREMENT_LLM", ""))
    parser.add_argument("--moderator-llm", default=os.environ.get("MODERATOR_LLM", ""))
    parser.add_argument("--doctor-image-request", default=os.environ.get("DOCTOR_IMAGE_REQUEST", "False"))
    parser.add_argument(
        "--harm-judge-mode",
        default="heuristic",
        choices=("off", "heuristic", "llm", "hybrid"),
    )
    parser.add_argument(
        "--harm-judge-model",
        default="ollama:llama3.1:8b",
    )
    parser.add_argument(
        "--refresh-dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate MPIB-V1 case index, manifest, and patient scripts before running.",
    )
    parser.add_argument(
        "--run-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Execute the MPIB-V1 batch runner before summarizing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    medsim_root, generator, runner, summarizer = _repo_paths()
    reports_root = Path(args.reports_root).resolve()
    reports_root.mkdir(parents=True, exist_ok=True)

    defenses = [item.strip() for item in args.defenses.split(",") if item.strip()]
    if not defenses:
        raise ValueError("At least one defense must be specified.")

    rules = parse_csv(args.rules, VALID_RULES) if args.rules.strip() else []
    timings = parse_csv(args.timings, VALID_TIMINGS) if args.timings.strip() else []
    tiers = parse_csv(args.tiers, VALID_TIERS) if args.tiers.strip() else []
    scenario_families = (
        parse_csv(args.scenario_families, VALID_SCENARIO_FAMILIES)
        if args.scenario_families.strip()
        else []
    )
    explicit_scenarios = (
        {int(item) for item in parse_csv(args.scenarios)}
        if args.scenarios.strip()
        else None
    )

    manifest_path = Path(args.manifest).resolve()
    script_input_dir = Path(args.script_input_dir).resolve()
    baseline_script_dir = Path(args.baseline_script_dir).resolve()
    case_index_path = Path(args.case_index).resolve()

    if args.refresh_dataset:
        generator_cmd = [
            sys.executable,
            str(generator),
            "all",
            "--source-cases",
            str(Path(args.source_cases).resolve()),
            "--medqa-jsonl",
            str(Path(args.medqa_jsonl).resolve()),
            "--output-case-index",
            str(case_index_path),
            "--output-manifest",
            str(manifest_path),
            "--output-script-dir",
            str(script_input_dir),
            "--rules",
            args.rules or "R1,R2,R4,R6",
            "--timings",
            args.timings or "early,late",
            "--tiers",
            args.tiers or "strict",
        ]
        if args.scenario_families.strip():
            generator_cmd += ["--scenario-families", args.scenario_families]
        rc = run_cmd(generator_cmd, cwd=medsim_root, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing MPIB-V1 manifest: {manifest_path}")

    manifest_rows = load_manifest_rows(manifest_path)
    selected_manifest_rows = filter_manifest_rows(
        manifest_rows,
        rules=rules,
        timings=timings,
        tiers=tiers,
        scenario_families=scenario_families,
        scenarios=explicit_scenarios,
    )
    if not selected_manifest_rows:
        raise RuntimeError("No MPIB-V1 manifest rows match the requested filters.")

    selected_scenarios = sorted({int(row["scenario_id"]) for row in selected_manifest_rows})
    if args.preset == "smoke":
        selected_scenarios = selected_scenarios[:5]
    if not selected_scenarios:
        raise RuntimeError("Scenario selection is empty after applying the preset.")

    selected_scenarios_csv = ",".join(str(sid) for sid in selected_scenarios)
    num_batches = math.ceil(len(selected_scenarios) / args.batch_size)

    for defense in defenses:
        output_root = reports_root / "runs" / defense
        if args.run_benchmark:
            for batch_index in range(num_batches):
                argv = [
                    sys.executable,
                    str(runner),
                    "--manifest",
                    str(manifest_path),
                    "--script-input-dir",
                    str(script_input_dir),
                    "--baseline-script-dir",
                    str(baseline_script_dir),
                    "--output-root",
                    str(output_root),
                    "--batch-index",
                    str(batch_index),
                    "--batch-size",
                    str(args.batch_size),
                    "--total-inferences",
                    str(args.total_inferences),
                    "--doctor-llm",
                    args.doctor_llm,
                    "--measurement-llm",
                    args.measurement_llm or args.doctor_llm,
                    "--moderator-llm",
                    args.moderator_llm or args.doctor_llm,
                    "--doctor-image-request",
                    args.doctor_image_request,
                    "--prompt-injection-defense",
                    defense,
                    "--scenarios",
                    selected_scenarios_csv,
                ]
                if args.rules.strip():
                    argv += ["--rules", args.rules]
                if args.timings.strip():
                    argv += ["--timings", args.timings]
                if args.tiers.strip():
                    argv += ["--tiers", args.tiers]
                if args.scenario_families.strip():
                    argv += ["--scenario-families", args.scenario_families]
                if args.dry_run:
                    argv.append("--dry-run")
                rc = run_cmd(argv, cwd=medsim_root, dry_run=False)
                if rc != 0:
                    return rc

        summary_json = reports_root / "summaries" / f"mpib_v1_{defense}.json"
        summary_csv_dir = reports_root / "csv" / defense
        summarize_cmd = [
            sys.executable,
            str(summarizer),
            "--root",
            str(output_root),
            "--manifest",
            str(manifest_path),
            "--output-json",
            str(summary_json),
            "--output-csv-dir",
            str(summary_csv_dir),
            "--harm-judge-mode",
            args.harm_judge_mode,
            "--harm-judge-model",
            args.harm_judge_model,
        ]
        if args.rules.strip():
            summarize_cmd += ["--rules", args.rules]
        if args.timings.strip():
            summarize_cmd += ["--timings", args.timings]
        if args.tiers.strip():
            summarize_cmd += ["--tiers", args.tiers]
        if args.scenario_families.strip():
            summarize_cmd += ["--scenario-families", args.scenario_families]
        summarize_cmd += ["--scenarios", selected_scenarios_csv]
        rc = run_cmd(summarize_cmd, cwd=medsim_root, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if args.dry_run:
        print(
            json.dumps(
                {
                    "reports_root": str(reports_root),
                    "manifest": str(manifest_path),
                    "selected_scenarios": selected_scenarios,
                    "defenses": defenses,
                    "rules": rules,
                    "timings": timings,
                    "tiers": tiers,
                    "scenario_families": scenario_families,
                    "note": "Dry run only: comparison artifacts are not generated because benchmark outputs are not written.",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    overall_rows: list[dict[str, Any]] = []
    by_rule_rows: list[dict[str, Any]] = []
    for defense in defenses:
        summary_path = reports_root / "summaries" / f"mpib_v1_{defense}.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing MPIB-V1 summary: {summary_path}")
        payload = load_json(summary_path)
        overall = list(((payload.get("summaries") or {}).get("overall_by_defense")) or [])
        if overall:
            row = dict(overall[0])
            row["summary_json"] = str(summary_path)
            overall_rows.append(row)
        for row in list(((payload.get("summaries") or {}).get("by_rule_timing_tier_defense")) or []):
            enriched = dict(row)
            enriched["summary_json"] = str(summary_path)
            by_rule_rows.append(enriched)

    overall_rows.sort(
        key=lambda row: (
            -float(row.get("attack_accuracy", 0.0) or 0.0),
            float(row.get("target_match_rate", 1.0) or 1.0),
            float(row.get("diagnosis_change_rate", 1.0) or 1.0),
            str(row.get("defense", "")),
        )
    )
    by_rule_rows.sort(
        key=lambda row: (
            str(row.get("rule_family_id", "")),
            str(row.get("tier", "")),
            str(row.get("timing", "")),
            -float(row.get("attack_accuracy", 0.0) or 0.0),
            str(row.get("defense", "")),
        )
    )

    comparison_dir = reports_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_csv(comparison_dir / "mpib_v1_overall_comparison.csv", overall_rows)
    write_csv(comparison_dir / "mpib_v1_rule_comparison.csv", by_rule_rows)

    summary_payload = {
        "reports_root": str(reports_root),
        "manifest": str(manifest_path),
        "selected_scenarios": selected_scenarios,
        "defenses": defenses,
        "rules": rules,
        "timings": timings,
        "tiers": tiers,
        "scenario_families": scenario_families,
        "comparison_files": {
            "overall": str(comparison_dir / "mpib_v1_overall_comparison.csv"),
            "by_rule": str(comparison_dir / "mpib_v1_rule_comparison.csv"),
            "report_md": str(comparison_dir / "mpib_v1_defense_report.md"),
        },
        "overall_ranked_rows": overall_rows,
        "by_rule_rows": by_rule_rows,
        "notes": {
            "attack_source": "MPIB-V1 manifest/scripts only",
            "sorting": "overall rows sorted by attack_accuracy desc, then target_match_rate asc, then diagnosis_change_rate asc",
        },
    }
    (comparison_dir / "mpib_v1_benchmark_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (comparison_dir / "mpib_v1_defense_report.md").write_text(
        build_markdown_report(
            overall_rows=overall_rows,
            by_rule_rows=by_rule_rows,
            suite_root=reports_root,
            manifest_path=manifest_path,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
