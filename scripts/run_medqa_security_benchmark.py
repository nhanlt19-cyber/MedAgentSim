#!/usr/bin/env python3
"""
Unified MedQA -> MedAgentSim security benchmark runner.

This script standardizes the benchmark matrix used in this workspace:
- dataset: MedQA
- runtime: MedAgentSim
- patient-channel prompt injection via OpenPI-style scripted inputs
- optional ASB-style observation poisoning via Measurement outputs
- standardized summaries and benchmark spec artifacts
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


VALID_SURFACES = ("patient", "observation")
VALID_ATTACKS = ("naive", "ignore", "escape", "fake_comp", "combine")
VALID_TIMINGS = ("early", "late")
DEFAULT_METRICS = (
    "PNA_T",
    "PNA_I",
    "ASV",
    "MR",
    "FPR",
    "FNR",
    "baseline_accuracy",
    "attack_accuracy",
    "accuracy_drop",
    "diagnosis_change_rate",
)


def _repo_paths() -> tuple[Path, Path, Path, Path, Path]:
    script = Path(__file__).resolve()
    medsim_root = script.parents[1]
    workspace_root = medsim_root.parent
    bridge = workspace_root / "attack" / "run_medqa_openpi_bridge.py"
    patient_runner = medsim_root / "scripts" / "run_medqa_openpi_batch_resume.py"
    observation_runner = medsim_root / "scripts" / "run_medqa_asb_observation_batch_resume.py"
    summarizer = medsim_root / "scripts" / "summarize_medqa_openpi_results.py"
    return medsim_root, workspace_root, bridge, patient_runner, observation_runner


def parse_csv(raw: str, valid: tuple[str, ...]) -> list[str]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in items if x not in valid]
    if bad:
        raise ValueError(f"Invalid values: {bad}; allowed: {valid}")
    return items


def parse_optional_csv(raw: str | None, valid: tuple[str, ...]) -> list[str]:
    if raw is None or not raw.strip():
        return []
    return parse_csv(raw, valid)


def run_cmd(argv: list[str], cwd: Path, env: dict[str, str] | None = None, dry_run: bool = False) -> int:
    print("+", " ".join(argv), flush=True)
    if dry_run:
        return 0
    result = subprocess.run(argv, cwd=str(cwd), env=env)
    return int(result.returncode)


def ensure_manifest(args: argparse.Namespace, workspace_root: Path, bridge_script: Path) -> int:
    manifest = Path(args.cases_file).resolve()
    if manifest.exists() and not args.refresh_manifest:
        return 0
    argv = [
        sys.executable,
        str(bridge_script),
        "build-manifest-medqa",
        "--medqa-jsonl",
        str(Path(args.medqa_jsonl).resolve()),
        "--output",
        str(manifest),
    ]
    if args.merge_from.strip():
        argv += ["--merge-from", str(Path(args.merge_from).resolve())]
    return run_cmd(argv, cwd=workspace_root, dry_run=args.dry_run)


def load_case_ids(cases_file: Path) -> list[int]:
    with cases_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Invalid cases file: {cases_file}")
    ids = [int(case["scenario_id"]) for case in cases if isinstance(case, dict) and "scenario_id" in case]
    ids.sort()
    return ids


def batch_indexes(total_items: int, batch_size: int) -> list[int]:
    return list(range(math.ceil(total_items / batch_size)))


def output_root_for(surface: str, defense: str, reports_root: Path) -> Path:
    return reports_root / "runs" / surface / defense


def summary_path_for(surface: str, defense: str, reports_root: Path) -> Path:
    return reports_root / "summaries" / f"{surface}_{defense}.json"


def write_benchmark_spec(
    args: argparse.Namespace,
    reports_root: Path,
    scenario_ids: list[int],
    surfaces: list[str],
    defenses: list[str],
    attacks: list[str],
    timings: list[str],
) -> None:
    payload = {
        "dataset": "MedQA",
        "runtime": "MedAgentSim",
        "cases_file": str(Path(args.cases_file).resolve()),
        "medqa_jsonl": str(Path(args.medqa_jsonl).resolve()),
        "scenario_ids": scenario_ids,
        "preset": args.preset,
        "surfaces": surfaces,
        "defenses": defenses,
        "attacks": attacks,
        "timings": timings,
        "include_injected_only": bool(args.include_injected_only),
        "doctor_llm": args.doctor_llm,
        "measurement_llm": args.measurement_llm or args.doctor_llm,
        "moderator_llm": args.moderator_llm or args.doctor_llm,
        "total_inferences": args.total_inferences,
        "metrics": list(DEFAULT_METRICS),
        "notes": [
            "patient surface uses existing OpenPI-style scripted patient injections",
            "observation surface poisons Measurement -> Doctor observations at runtime",
            "PNA-I and MR are only meaningful for patient surface runs with injected-only traces",
        ],
    }
    spec_path = reports_root / "benchmark_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_surface(
    medsim_root: Path,
    summarizer: Path,
    root: Path,
    surface: str,
    args: argparse.Namespace,
    output_json: Path,
) -> int:
    argv = [
        sys.executable,
        str(summarizer),
        "--root",
        str(root),
        "--cases-file",
        str(Path(args.cases_file).resolve()),
        "--script-input-dir",
        str(Path(args.script_input_dir).resolve()),
        "--attacks",
        args.attacks,
        "--timings",
        args.timings,
        "--surface",
        surface,
        "--output-json",
        str(output_json),
        "--no-show-openpi-metrics",
    ]
    return run_cmd(argv, cwd=medsim_root, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    medsim_root, _, _, _, _ = _repo_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="smoke", choices=("smoke", "full"))
    parser.add_argument("--surfaces", default="patient,observation", help="Comma-separated surfaces: patient, observation")
    parser.add_argument("--defenses", default="none,layered_guard", help="Comma-separated doctor defense modes to benchmark.")
    parser.add_argument("--attacks", default=",".join(VALID_ATTACKS))
    parser.add_argument("--timings", default="early,late")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--total-inferences", type=int, default=10)
    parser.add_argument(
        "--cases-file",
        default=str(medsim_root / "scripted_inputs_medqa" / "medqa_all_107_cases.json"),
    )
    parser.add_argument(
        "--medqa-jsonl",
        default=str(medsim_root / "datasets" / "_medqa.jsonl"),
    )
    parser.add_argument(
        "--merge-from",
        default=str(medsim_root / "scripted_inputs_medqa" / "medqa_benchmark_cases.json"),
        help="Optional handcrafted manifest overlay for build-manifest-medqa.",
    )
    parser.add_argument(
        "--script-input-dir",
        default=str(medsim_root / "scripted_inputs_medqa"),
    )
    parser.add_argument(
        "--reports-root",
        default=str(medsim_root / "output_eval_medqa_security"),
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
        help="Generate and run injected-only patient scripts for true PNA-I / MR on the patient surface.",
    )
    parser.add_argument(
        "--refresh-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rebuild the MedQA cases manifest before running.",
    )
    parser.add_argument(
        "--run-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Execute the batch runners. Disable to summarize existing outputs only.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    medsim_root, workspace_root, bridge_script, patient_runner, observation_runner = _repo_paths()
    summarizer = medsim_root / "scripts" / "summarize_medqa_openpi_results.py"
    reports_root = Path(args.reports_root).resolve()
    reports_root.mkdir(parents=True, exist_ok=True)

    surfaces = parse_csv(args.surfaces, VALID_SURFACES)
    attacks = parse_csv(args.attacks, VALID_ATTACKS)
    timings = parse_csv(args.timings, VALID_TIMINGS)
    defenses = [item.strip() for item in args.defenses.split(",") if item.strip()]
    if not defenses:
        raise ValueError("At least one defense must be specified in --defenses")

    rc = ensure_manifest(args, workspace_root, bridge_script)
    if rc != 0:
        return rc

    scenario_ids = load_case_ids(Path(args.cases_file).resolve())
    if args.preset == "smoke":
        scenario_ids = scenario_ids[:5]
    batch_ids = batch_indexes(len(scenario_ids), args.batch_size)
    write_benchmark_spec(args, reports_root, scenario_ids, surfaces, defenses, attacks, timings)

    env = os.environ.copy()
    for surface in surfaces:
        runner = patient_runner if surface == "patient" else observation_runner
        for defense in defenses:
            output_root = output_root_for(surface, defense, reports_root)
            if args.run_benchmark:
                for batch_index in batch_ids:
                    argv = [
                        sys.executable,
                        str(runner),
                        "--cases-file",
                        str(Path(args.cases_file).resolve()),
                        "--script-input-dir",
                        str(Path(args.script_input_dir).resolve()),
                        "--output-root",
                        str(output_root),
                        "--batch-index",
                        str(batch_index),
                        "--batch-size",
                        str(args.batch_size),
                        "--attacks",
                        ",".join(attacks),
                        "--timings",
                        ",".join(timings),
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
                    ]
                    if surface == "patient" and args.include_injected_only:
                        argv.append("--include-injected-only")
                    if args.global_target.strip():
                        argv += ["--global-target", args.global_target.strip()]
                    rc = run_cmd(argv, cwd=medsim_root, env=env, dry_run=args.dry_run)
                    if rc != 0:
                        return rc

            output_json = summary_path_for(surface, defense, reports_root)
            rc = summarize_surface(medsim_root, summarizer, output_root, surface, args, output_json)
            if rc != 0:
                return rc

    manifest = {
        "reports_root": str(reports_root),
        "scenarios": scenario_ids,
        "surfaces": surfaces,
        "defenses": defenses,
        "attacks": attacks,
        "timings": timings,
        "summary_files": {
            surface: {
                defense: str(summary_path_for(surface, defense, reports_root))
                for defense in defenses
            }
            for surface in surfaces
        },
    }
    (reports_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
