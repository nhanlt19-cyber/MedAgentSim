#!/usr/bin/env python3
"""
Unified ASB-style benchmark runner for MedAgentSim on MedQA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


VALID_FAMILIES = ("clean", "dpi", "opi", "memory", "mixed", "pot_backdoor")


def _repo_paths() -> tuple[Path, Path, Path]:
    script = Path(__file__).resolve()
    medsim_root = script.parents[1]
    batch_runner = medsim_root / "scripts" / "run_medqa_asb_attack_batch_resume.py"
    summarizer = medsim_root / "scripts" / "summarize_medqa_asb_results.py"
    return medsim_root, batch_runner, summarizer


def run_cmd(argv: list[str], cwd: Path, dry_run: bool = False) -> int:
    print("+", " ".join(argv), flush=True)
    if dry_run:
        return 0
    result = subprocess.run(argv, cwd=str(cwd), env=os.environ.copy())
    return int(result.returncode)


def load_case_count(cases_file: Path) -> int:
    with cases_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Invalid cases file: {cases_file}")
    return len(cases)


def build_parser() -> argparse.ArgumentParser:
    medsim_root, _, _ = _repo_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="smoke", choices=("smoke", "full"))
    parser.add_argument("--families", default="dpi,opi,memory,mixed,pot_backdoor")
    parser.add_argument("--defenses", default="none,layered_guard")
    parser.add_argument("--attacks", default="naive,ignore,escape,fake_comp,combine")
    parser.add_argument("--timings", default="early,late")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--total-inferences", type=int, default=10)
    parser.add_argument("--doctor-llm", default=os.environ.get("DOCTOR_LLM", os.environ.get("REMOTE_LLM_MODEL", "Qwen3.5-27B-Q4_K_M.gguf")))
    parser.add_argument("--measurement-llm", default=os.environ.get("MEASUREMENT_LLM", ""))
    parser.add_argument("--moderator-llm", default=os.environ.get("MODERATOR_LLM", ""))
    parser.add_argument("--doctor-image-request", default=os.environ.get("DOCTOR_IMAGE_REQUEST", "False"))
    parser.add_argument("--global-target", default=os.environ.get("GLOBAL_TARGET", ""))
    parser.add_argument("--pot-trigger", default=os.environ.get("POT_BACKDOOR_TRIGGER", "with perspicacious discernment"))
    parser.add_argument(
        "--cases-file",
        default=str(medsim_root / "scripted_inputs_medqa" / "medqa_all_107_cases.json"),
    )
    parser.add_argument(
        "--reports-root",
        default=str(medsim_root / "output_eval_medqa_asb"),
    )
    parser.add_argument(
        "--run-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    medsim_root, batch_runner, summarizer = _repo_paths()
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    defenses = [item.strip() for item in args.defenses.split(",") if item.strip()]
    reports_root = Path(args.reports_root).resolve()
    reports_root.mkdir(parents=True, exist_ok=True)
    cases_file = Path(args.cases_file).resolve()

    total_cases = load_case_count(cases_file)
    if args.preset == "smoke":
        total_cases = min(total_cases, 5)
    num_batches = math.ceil(total_cases / args.batch_size)

    for family in families:
        if family not in VALID_FAMILIES:
            raise ValueError(f"Unsupported family {family!r}; allowed: {VALID_FAMILIES}")

    for defense in defenses:
        defense_root = reports_root / "runs" / defense
        defense_root.mkdir(parents=True, exist_ok=True)
        if args.run_benchmark:
            for family in ("clean", *families):
                for batch_index in range(num_batches):
                    argv = [
                        sys.executable,
                        str(batch_runner),
                        "--family",
                        family,
                        "--cases-file",
                        str(cases_file),
                        "--output-root",
                        str(defense_root),
                        "--batch-index",
                        str(batch_index),
                        "--batch-size",
                        str(args.batch_size),
                        "--attacks",
                        args.attacks,
                        "--timings",
                        args.timings,
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
                        "--global-target",
                        args.global_target,
                        "--pot-trigger",
                        args.pot_trigger,
                    ]
                    rc = run_cmd(argv, cwd=medsim_root, dry_run=args.dry_run)
                    if rc != 0:
                        return rc

        summary_json = reports_root / "summaries" / f"asb_{defense}.json"
        summary_csv_dir = reports_root / "csv" / defense
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        rc = run_cmd(
            [
                sys.executable,
                str(summarizer),
                "--root",
                str(defense_root),
                "--output-json",
                str(summary_json),
                "--output-csv-dir",
                str(summary_csv_dir),
            ],
            cwd=medsim_root,
            dry_run=args.dry_run,
        )
        if rc != 0:
            return rc

    manifest = {
        "reports_root": str(reports_root),
        "cases_file": str(cases_file),
        "preset": args.preset,
        "families": families,
        "defenses": defenses,
        "attacks": [item.strip() for item in args.attacks.split(",") if item.strip()],
        "timings": [item.strip() for item in args.timings.split(",") if item.strip()],
        "summary_files": {
            defense: str(reports_root / "summaries" / f"asb_{defense}.json")
            for defense in defenses
        },
    }
    (reports_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
